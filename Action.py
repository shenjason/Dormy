"""One thing Johnny can do, described well enough for Gemini to decide to do it.

The design question this answers is "how does the model learn what my functions are, and
how do the arguments get to me without being spoken out loud?" -- and the answer is that
the Live API already keeps those on two separate channels:

    audio / transcript      "Sure, making the desk lights blue."
    tool channel            set_light_color(zone="desk", color="blue")

`LiveServerMessage.tool_call` is a *sibling* of `server_content`, not part of it, and it
carries `args` as structured JSON. Function calls never appear in
`output_audio_transcription`. So nothing here parses a transcript, and nothing makes the
model read arguments aloud -- see ActionManager.briefing() for the one nudge that keeps the
spoken half sounding human.

Writing an action is a one-liner, because the parameter schema is inferred from the
function's own signature:

    def set_light_color(zone: Literal["desk", "bed"], color: str, brightness: int = 100):
        ...

    Action("Set the colour of an LED zone. Use when asked to change the lights.",
           set_light_color)

which produces zone (string, enum desk|bed, required), color (string, required) and
brightness (integer, optional). Subclass instead when an action needs state -- override
run() and pass no function; the signature of run() is read the same way.

The `purpose` string is the highest-leverage field in the whole file. It is the only thing
the model reads when deciding *whether* to call this action, so write it as an instruction
about when to use it, not as a label.
"""

import inspect
import types as pytypes      # stdlib; `types` below is google.genai.types
import typing

from google.genai import types

# Python annotation -> Gemini schema type. Anything not in here is handled specially
# (Literal, list) or falls back to STRING with a warning, because a wrong type silently
# produces arguments the handler cannot accept.
_SCALARS = {
    str: types.Type.STRING,
    int: types.Type.INTEGER,
    float: types.Type.NUMBER,
    bool: types.Type.BOOLEAN,
}


class ActionError(Exception):
    """Raised at construction time. Always a bug in the action, never a runtime failure."""


def _unwrap_optional(hint):
    """Optional[X] / X | None -> X. The model has `required` to express optionality."""
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is getattr(pytypes, "UnionType", None):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return hint


def _schema_for(hint, description, action_name, param_name):
    """One parameter's types.Schema, inferred from its annotation."""
    hint = _unwrap_optional(hint)
    origin = typing.get_origin(hint)

    # Literal["desk", "bed"] -> a string with an enum. This is the highest-value case:
    # without it the model happily invents a zone name that no handler knows.
    if origin is typing.Literal:
        values = [str(v) for v in typing.get_args(hint)]
        return types.Schema(
            type=types.Type.STRING, enum=values, description=description
        )

    if origin in (list, typing.List):
        args = typing.get_args(hint)
        item = args[0] if args else str
        return types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=_SCALARS.get(item, types.Type.STRING)),
            description=description,
        )

    if hint in _SCALARS:
        return types.Schema(type=_SCALARS[hint], description=description)

    if hint is inspect.Parameter.empty:
        print(
            f"  ! {action_name}({param_name}): no type hint, assuming string. "
            "Annotate it -- the model picks argument types from this schema."
        )
    else:
        print(
            f"  ! {action_name}({param_name}): unsupported annotation {hint!r}, "
            "assuming string."
        )
    return types.Schema(type=types.Type.STRING, description=description)


class Action:
    """A named capability: what it is for, and the function that carries it out."""

    def __init__(self, purpose, function=None, name=None, describe=None):
        """
        purpose   what this does and *when* to use it. Gemini reads this to decide.
        function  the handler. Omit it in a subclass that overrides run().
        name      defaults to the function's own name; must be unique in a manager.
        describe  {parameter: "text"} for wording a type hint cannot carry, e.g.
                  {"color": "a CSS colour name such as 'warm white'"}.
        """
        if not purpose or not purpose.strip():
            raise ActionError("an Action needs a purpose -- it is what the model reads")

        self.purpose = purpose.strip()
        self.function = function
        self.describe = dict(describe or {})

        target = self._target()
        self.name = name or getattr(target, "__name__", type(self).__name__)
        # The API's own constraint on function names.
        if not self.name.replace("_", "").replace("-", "").isalnum():
            raise ActionError(f"action name {self.name!r} must be alphanumeric plus _ -")

        self.signature = inspect.signature(target)
        self.is_async = inspect.iscoroutinefunction(target)
        self._parameters = self._build_parameters(target)

        unknown = set(self.describe) - set(self.signature.parameters)
        if unknown:
            raise ActionError(
                f"{self.name}: describe names parameters that do not exist: "
                f"{', '.join(sorted(unknown))}"
            )

    # --- construction helpers ---

    def _target(self):
        """The callable whose signature defines this action's parameters."""
        if self.function is not None:
            if not callable(self.function):
                raise ActionError(f"{self.function!r} is not callable")
            return self.function
        if type(self).run is Action.run:
            raise ActionError(
                "an Action needs either a function or a subclass that overrides run()"
            )
        return self.run          # bound, so `self` is already out of the signature

    def _build_parameters(self, target):
        """types.Schema for the whole argument object, or None when there are none."""
        try:
            hints = typing.get_type_hints(target)
        except Exception:
            hints = {}           # forward refs we cannot resolve: fall back per-parameter

        properties = {}
        required = []
        for param_name, param in self.signature.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                # *args / **kwargs cannot be described to the model, so they can never
                # be filled -- better to say so now than to see empty calls later.
                raise ActionError(
                    f"{self.name}: *args/**kwargs cannot be described to the model"
                )
            properties[param_name] = _schema_for(
                hints.get(param_name, param.annotation),
                self.describe.get(param_name),
                self.name,
                param_name,
            )
            if param.default is param.empty:
                required.append(param_name)

        if not properties:
            # A no-argument function: omit `parameters` entirely rather than sending an
            # empty object, which some models treat as "one unnamed argument".
            return None
        return types.Schema(
            type=types.Type.OBJECT, properties=properties, required=required or None
        )

    # --- what the manager needs ---

    def declaration(self):
        """The types.FunctionDeclaration that teaches Gemini this action exists."""
        return types.FunctionDeclaration(
            name=self.name,
            description=self.purpose,
            parameters=self._parameters,
        )

    def summary(self):
        """One line for the briefing text and for startup logging."""
        params = ", ".join(self.signature.parameters) or ""
        return f"{self.name}({params}) -- {self.purpose}"

    # --- running ---

    def run(self, **args):
        """Carry the action out. Override in a subclass, or pass a function."""
        return self.function(**args)

    def invoke(self, args):
        """run() with the model's arguments, normalised into a response dict.

        Never raises. A handler that blows up has to reach Gemini as an error it can
        explain out loud -- "Spotify isn't linked yet" -- rather than taking the
        conversation down with it.
        """
        try:
            bound = self.signature.bind(**(args or {}))
            bound.apply_defaults()
        except TypeError as exc:
            # The model invented or omitted an argument. Telling it exactly what was
            # wrong is usually enough for it to retry correctly.
            return {"error": f"bad arguments for {self.name}: {exc}"}

        try:
            result = self.run(**bound.arguments)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        return self.normalise(result)

    @staticmethod
    def normalise(result):
        """Whatever the handler returned -> the dict send_tool_response wants.

        The key is "output" because that is the one the API documents:
        FunctionResponse.response is 'Use "output" key to specify function output and
        "error" key to specify error details'. A dict from the handler is passed through
        untouched on the assumption it already speaks that convention.
        """
        if result is None:
            return {"output": "ok"}
        if isinstance(result, dict):
            return result
        return {"output": result}

    def __repr__(self):
        return f"<Action {self.summary()}>"
