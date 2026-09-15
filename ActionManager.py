"""The registry of Actions, and the bridge between Gemini's tool channel and them.

Three jobs:

  tools()      turn every registered Action into the function declarations that go into
               LiveConnectConfig.tools before a conversation opens.
  briefing()   a short block of text appended to the system instruction.
  dispatch()   receive a LiveServerMessage.tool_call, run the handlers, and send the
               results back with session.send_tool_response().

On the briefing, because it is not what it first looks like: Gemini does not learn the
tools from prose. It learns them from the declarations, which travel as structured data --
name, description, typed parameters. A concatenated wall of text describing the same
functions would be redundant at best and contradictory at worst. What text *is* good for
is policy: which zones exist, and how to talk while using them. That is what briefing()
carries, and it is deliberately short.

Run this file directly for a self-test that needs no API key and no hardware:

    .venv/bin/python ActionManager.py
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Literal
import openmeteo_requests
import Spotify
from ESP32DeviceManager import ESP32DeviceManager as ESP

from google.genai import types

from Action import Action, ActionError

# How long a handler may run before dispatch gives up on it and reports a timeout to
# Gemini. Every action is blocking -- the model waits for the result before it speaks --
# so this is also the longest the conversation can stall on one. The worker thread is not
# killed when this fires (nothing in Python can), it is only stopped from being waited on;
# a handler that hangs forever leaks one executor thread.
ACTION_TIMEOUT = 15.0


class ActionManager:
    """Holds the Actions, briefs Gemini on them, and runs the calls that come back."""

    def __init__(self, actions=()):
        self.actions = {}
        self.add_all(actions)
        # Ids cancelled by the server while their handler was still running. A barge-in
        # cancels the turn that asked, so answering it afterwards would push a stale
        # result into a conversation that has moved on.
        self._cancelled = set()


        
    
    # --- registry ---

    def add(self, action):
        if not isinstance(action, Action):
            raise ActionError(f"{action!r} is not an Action")
        if action.name in self.actions:
            # Two actions with one name means the second silently shadows the first in
            # the model's tool list, and calls go to whichever won. Never worth allowing.
            raise ActionError(f"duplicate action name: {action.name}")
        self.actions[action.name] = action
        return action

    def add_all(self, actions):
        for action in actions:
            self.add(action)
        return self

    def __len__(self):
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions.values())

    def __contains__(self, name):
        return name in self.actions

    # --- what Gemini is told ---

    def tools(self):
        """The value for LiveConnectConfig.tools. Empty list when nothing is registered."""
        if not self.actions:
            return []
        return [
            types.Tool(
                function_declarations=[a.declaration() for a in self.actions.values()]
            )
        ]

    def briefing(self):
        """Policy text for the system instruction. Not a substitute for the declarations."""
        if not self.actions:
            return ""
        names = ", ".join(self.actions)
        return (
            f"\n\nYou can act on the room through these functions: {names}. "
            "Their descriptions and arguments are given to you separately -- use them "
            "whenever the user asks for something one of them does, rather than saying "
            "you are unable to. "
            # The one line that keeps the spoken half human. The arguments already reach
            # the handler on their own channel, so speaking them is pure noise.
            "When you call one, speak like a person: acknowledge in a few natural words "
            "and then stop. Never read out function names, argument names or argument "
            "values, never announce that you are calling a tool, and never describe the "
            "result in a structured way. Say \"Okay, desk lights blue.\", not "
            "\"Calling set_light_color with zone desk and color blue.\""
        )

    def describe(self):
        """Multi-line inventory, for printing at startup."""
        if not self.actions:
            return "  (none registered)"
        return "\n".join(f"  {a.summary()}" for a in self.actions.values())

    # --- running the calls ---

    def cancel(self, ids):
        """Handle LiveServerMessage.tool_call_cancellation."""
        new = set(ids or ()) - self._cancelled
        if new:
            self._cancelled.update(new)
            print(f"  (tool call cancelled: {', '.join(sorted(new))})")

    async def dispatch(self, session, tool_call, on_call=None):
        """Run every function call in `tool_call` and send the responses back.

        `on_call(name, args, response)` is invoked per call, for state only the caller
        owns -- testChatWakeup.py uses it to notice end_conversation.

        Handlers run in the default executor, NOT inline. This is not stylistic: the
        uplink pump that feeds the microphone to Gemini lives on this same event loop,
        so a handler that blocks for two seconds on a subprocess stops the mic for two
        seconds. The model goes deaf mid-sentence and it presents as an audio fault, not
        as a slow action.
        """
        loop = asyncio.get_running_loop()
        responses = []

        for call in tool_call.function_calls or []:
            action = self.actions.get(call.name)
            args = dict(call.args or {})

            if action is None:
                print(f"  [action] {call.name}(...) -> unknown")
                response = {"error": f"unknown function {call.name}"}
            else:
                started = time.perf_counter()
                try:
                    response = await asyncio.wait_for(
                        self._run(loop, action, args), timeout=ACTION_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    response = {
                        "error": f"{action.name} timed out after {ACTION_TIMEOUT:.0f}s"
                    }
                shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
                # Printed so the tool channel is visible while the audio channel stays
                # clean -- the two really are separate, and this is the proof.
                print(
                    f"  [action] {action.name}({shown}) -> {response} "
                    f"[{time.perf_counter() - started:.2f}s]"
                )

            if on_call is not None:
                on_call(call.name, args, response)

            if call.id in self._cancelled:
                self._cancelled.discard(call.id)
                continue

            responses.append(
                types.FunctionResponse(id=call.id, name=call.name, response=response)
            )

        if responses:
            # id is mandatory on the Gemini API path; send_tool_response raises without
            # it. Sending them together is one round trip instead of several.
            await session.send_tool_response(function_responses=responses)
        return responses

    @staticmethod
    async def _run(loop, action, args):
        """One handler, on a worker thread unless it is already a coroutine."""
        if action.is_async:
            try:
                bound = action.signature.bind(**args)
                bound.apply_defaults()
            except TypeError as exc:
                return {"error": f"bad arguments for {action.name}: {exc}"}
            try:
                return Action.normalise(await action.run(**bound.arguments))
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}
        return await loop.run_in_executor(None, action.invoke, args)




# WMO weather codes, as words. The API returns a bare integer and Gemini has to say
# something out loud -- handing it "95" invites a guess, handing it "thunderstorm" does
# not. Same reasoning as the "output" key: give the model the fact, not a puzzle.
WEATHER_CODES = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "light freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "light snow showers", 86: "snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}

# Blacksburg, VA -- the dorm. Hardcoded on purpose: the model is given no `location`
# argument, because answering for an arbitrary city would need geocoding this action
# does not do, and a location it cannot honour is an invitation to hallucinate one.
LATITUDE = 37.224102
LONGITUDE = -80.418608
FORECAST_DAYS = 7


def _sky(code):
    return WEATHER_CODES.get(int(code), f"weather code {int(code)}")


def get_weather():
    """Current conditions plus a seven-day forecast for the dorm, as plain text.

    Returned as one string rather than a dict because the model reads it and then says
    one sentence of it out loud; the shape it needs is prose it can quote, not a record
    it has to navigate. Values are exact -- rounding them here only gives the model a
    less true number to speak, and it will round for itself when it talks.
    """
    openmeteo = openmeteo_requests.Client()

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": ["temperature_2m", "relative_humidity_2m", "weather_code"],
        "daily": ["temperature_2m_min", "temperature_2m_max", "precipitation_probability_max", "weather_code"],
        "timezone": "auto",
        "temperature_unit": "fahrenheit",
    }
    responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)

    response = responses[0]

    # Every timestamp the API hands back is a UTC epoch, including the ones it selected
    # with timezone=auto. Rendering them as UTC would have told the user it is 6 PM at
    # lunchtime, so the offset the response carries is applied by hand.
    local = timezone(timedelta(seconds=response.UtcOffsetSeconds()))
    current = response.Current()
    daily = response.Daily()

    lines = [
        f"Weather for the dorm (Blacksburg, VA), local time "
        f"{datetime.fromtimestamp(current.Time(), local):%A %-I:%M %p}.",
        f"Now: {current.Variables(0).Value():.0f} F, "
        f"{current.Variables(1).Value():.0f}% humidity, "
        f"{_sky(current.Variables(2).Value())}.",
        "Forecast:",
    ]

    temp_min = daily.Variables(0).ValuesAsNumpy()
    temp_max = daily.Variables(1).ValuesAsNumpy()
    precipitation = daily.Variables(2).ValuesAsNumpy()
    codes = daily.Variables(3).ValuesAsNumpy()
    day_start = daily.Time()
    interval = daily.Interval()

    for i in range(min(FORECAST_DAYS, len(temp_min))):
        # Named days, not "day 3". The user asks about Thursday, and a number would make
        # the model count -- off by one is the likeliest way this answer goes wrong.
        day = datetime.fromtimestamp(day_start + i * interval, local)
        label = "today" if i == 0 else "tomorrow" if i == 1 else f"{day:%A}"
        lines.append(
            f"  {label} ({day:%b %-d}): low {temp_min[i]:.0f} F, high {temp_max[i]:.0f} F, "
            f"{precipitation[i]:.0f}% chance of precipitation, {_sky(codes[i])}"
        )

    return "\n".join(lines)


def get_time():
    """The wall clock, spoken naturally rather than as a 24-hour string."""
    return datetime.now().strftime("%-I:%M %p on %A")


def cpu_temperature():
    """Real reading, so at least one action proves the *return* path end to end."""
    with open("/sys/class/thermal/thermal_zone0/temp") as handle:
        return round(int(handle.read().strip()) / 1000.0, 1)


def builtin_actions():
    """The default set. Purposes are written as instructions about *when* to call."""
    actions = [
        Action(
            "Tell the user the current time or day. Call this whenever the user asks "
            "what time or what day it is -- do not guess, you have no clock otherwise.",
            get_time,
        ),
        Action(
            "Read the Raspberry Pi's CPU temperature in degrees Celsius. Call this when "
            "asked how hot the Pi is, or whether it is overheating.",
            cpu_temperature,
        ),
        Action(
            "Look up the real weather for the dorm in Blacksburg, Virginia: conditions "
            "right now, and the low, high, chance of precipitation and sky for each of "
            "the next seven days. Call this for any question about the weather, the "
            "temperature outside, rain, snow or what to wear -- today or later this "
            "week -- and never answer from memory, because you have no other source and "
            "the forecast changes hourly. It returns the whole week at once, so read "
            "back only the day or two the user asked about.",
            get_weather,
        ),
    ]
    return actions + Spotify.actions()


# --- self-test -------------------------------------------------------------


# def _self_test():
    # """Everything the framework promises, with no API key and no hardware."""

    # print("=== inference ===")
    # manager = ActionManager(builtin_actions())
    # print(manager.describe())

    # print("\n=== declarations ===")
    # for action in manager:
    #     declaration = action.declaration()
    #     params = declaration.parameters
    #     if params is None:
    #         print(f"  {declaration.name}: no parameters")
    #         continue
    #     for name, schema in params.properties.items():
    #         required = name in (params.required or [])
    #         enum = f" enum={schema.enum}" if schema.enum else ""
    #         print(
    #             f"  {declaration.name}.{name}: {schema.type.value}"
    #             f"{enum} {'required' if required else 'optional'}"
    #         )

    # print("\n=== guards ===")
    # for label, thunk in (
    #     ("duplicate name", lambda: ActionManager([builtin_actions()[0]] * 2)),
    #     ("no purpose", lambda: Action("", get_time)),
    #     ("not callable", lambda: Action("x", 42)),
    #     ("bare Action", lambda: Action("x")),
    #     ("**kwargs", lambda: Action("x", lambda **kw: kw, name="k")),
    # ):
    #     try:
    #         thunk()
    #         print(f"  {label}: NOT CAUGHT")
    #     except ActionError as exc:
    #         print(f"  {label}: {exc}")

    # print("\n=== dispatch ===")

    # def explode():
    #     raise RuntimeError("the ESP32 is unplugged")

    # def slow():
    #     time.sleep(ACTION_TIMEOUT + 2)

    # manager.add(Action("Always fails, to prove errors reach Gemini.", explode))
    # manager.add(Action("Always times out.", slow))

    # class FakeSession:
    #     def __init__(self):
    #         self.sent = []

    #     async def send_tool_response(self, *, function_responses):
    #         self.sent.extend(function_responses)

    # def call(name, args=None, id_="id-1"):
    #     return types.LiveServerToolCall(
    #         function_calls=[types.FunctionCall(id=id_, name=name, args=args or {})]
    #     )

    # async def run():
    #     session = FakeSession()
    #     await manager.dispatch(session, call("get_time"))
    #     await manager.dispatch(session, call("cpu_temperature"))
    #     await manager.dispatch(
    #         session, call("set_light_color", {"zone": "desk", "color": "blue"})
    #     )
    #     await manager.dispatch(session, call("set_light_power", {"zone": "bed", "on": True}))
    #     await manager.dispatch(session, call("set_light_color", {"zone": "desk"}))
    #     await manager.dispatch(session, call("no_such_function"))
    #     await manager.dispatch(session, call("explode"))

    #     print("\n=== cancellation ===")
    #     manager.cancel(["id-9"])
    #     await manager.dispatch(session, call("get_time", id_="id-9"))
    #     print("  (no response sent for a cancelled id -- correct)")

    #     print(f"\n=== timeout ({ACTION_TIMEOUT:.0f}s, be patient) ===")
    #     await manager.dispatch(session, call("slow"))

    #     print(f"\nresponses sent: {len(session.sent)}")

    # asyncio.run(run())

    # print("\n=== briefing appended to the system instruction ===")
    # print(manager.briefing().strip())


if __name__ == "__main__":
    print(get_weather())

    # _self_test()
