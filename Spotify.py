"""Spotify actions, talking to the Web API directly through spotipy.

This replaced `spotify-cli`, which broke and could not be safely kept:

- Its `auth login` is dead on Python 3.13. It pulls in PyInquirer -> prompt_toolkit 1.0.14,
  which does `from collections import Mapping` -- removed in 3.10. Only the *scope picker*
  needed it; all 18 of its command modules import fine otherwise.
- The disqualifying part is where its auth goes. `cli/utils/Spotify.py:44-88` posts to
  `https://asia-east2-spotify-cli-283006.cloudfunctions.net/auth-refresh` on **every token
  refresh**, and if you supply your own application credentials it sends your **client secret**
  there too. That is the package author's personal cloud function, untouched since 2020. It is
  still answering, which makes it a worse trap, not a better one.

Here, tokens never leave this Pi: spotipy runs the OAuth flow against Spotify itself and caches
the refresh token in TOKEN_CACHE.

## Setup, once

1. Create an app at https://developer.spotify.com/dashboard. Any name. Add exactly this
   redirect URI:

       http://127.0.0.1:8888/callback

   It must be the literal loopback IP -- Spotify rejects `localhost` now, and plain http is
   allowed only for loopback.

2. Put the two values in `Dormy/.env`, next to GEMINI_API_KEY (the file is gitignored):

       SPOTIFY_CLIENT_ID=...
       SPOTIFY_CLIENT_SECRET=...

3. Authorise once. Two ways, because the redirect points at a port on *this* Pi while the
   browser is on your laptop:

       # a) tunnel that port over SSH, and the redirect completes by itself
       ssh -L 8888:127.0.0.1:8888 johnny@<pi>
       .venv/bin/python Spotify.py login --server

       # b) no tunnel: approve, then hand back the URL the browser was sent to
       .venv/bin/python Spotify.py login "http://127.0.0.1:8888/callback?code=..."

   (b) is fiddly for a reason worth knowing: the redirect goes to an address with nothing
   listening, so the tab hangs and looks broken -- and on a *second* authorisation Spotify
   skips the consent screen, so the tab appears never to load at all. The code is in the
   address bar either way. (a) avoids all of it: spotipy binds a real listener on
   127.0.0.1:8888 here, and the tunnel makes your laptop's 127.0.0.1:8888 reach it.

## What plays the music

`~/.config/systemd/user/librespot.service` makes this Pi a Spotify Connect device named
PI_DEVICE, with `PULSE_SINK=jarvis_aec_sink` so music becomes far-end audio the echo canceller
removes -- which is why the wake word still works over it, and why music is mono.

Appearing in the device list is not the same as being *active*, so `spotify_play` transfers
playback to the Pi and retries when nothing is playing anywhere. The other verbs deliberately
do not: "pause" with nothing playing should say so, not go hunting for a speaker.

## Choosing what to play

Spotify's search ranking is not stable across page sizes, and the small page is the bad one.
Measured here, 6 trials each and deterministic both ways:

    q="Moog City", limit=1    -> "Aria Math" by C418        6/6
    q="Moog City", limit=10   -> "Moog City" by C418        6/6

So the old `limit=1` search was not merely unlucky, it was asking for the worse ranking --
that is the whole of the `spotify_play(query="Moog City")` -> "Aria Math" bug. Everything now
asks for a page (`SEARCH_LIMIT`, which is also the real ceiling: >10 answers 400 here) and
re-ranks it locally with `_match_score`, so an exact title beats a more popular near-miss and
the answer no longer depends on Spotify's mood.

The score also decides *how much* to play. A query that matches a title is one track; a query
that matches nothing by name ("some jazz") is a mood, so the page is queued whole and the
music keeps going.

Playlists come from `/me/playlists` and are matched by the same scorer, which is why
`SCOPES` grew two entries -- see the note there about re-authorising.
"""

import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Literal

import spotipy
from spotipy.cache_handler import CacheFileHandler
from spotipy.oauth2 import SpotifyOAuth, start_local_http_server

from Action import Action

# Must match --name in ~/.config/systemd/user/librespot.service.
PI_DEVICE = "Johnny"

REDIRECT_URI = "http://127.0.0.1:8888/callback"

# Everything the actions below need and nothing more. Adding a scope invalidates the cached
# token, so the next call will ask you to authorise again -- spotipy's validate_token()
# rejects a cached token whose scope is not a superset of this string. The two playlist
# scopes arrived with spotify_playlist(); /me/playlists needs playlist-read-private even to
# see your own private playlists, and -collaborative for ones other people can edit.
SCOPES = (
    "user-read-playback-state user-modify-playback-state user-read-currently-playing "
    "playlist-read-private playlist-read-collaborative"
)

# Spotify's search ranking is *different* at limit=1 than at limit=5+, and worse: measured on
# this account, q="Moog City" returns "Aria Math" 6 times out of 6 at limit=1 and "Moog City"
# 6/6 at limit=10. Both are deterministic, so this is the ranker, not luck. Never search with
# limit=1 -- ask for a page and choose from it (_best_match below).
#
# 10 is also the ceiling in practice: 12, 15, 20, 25 and 50 all come back
# `400 Invalid limit` here, despite the documented maximum of 50.
SEARCH_LIMIT = 10

# The library playlist list is re-fetched at most this often; a name lookup that misses
# refreshes anyway, so a playlist made a minute ago is still findable.
PLAYLIST_TTL = 300

TOKEN_CACHE = Path.home() / ".config" / "dormy" / "spotify-token.json"
ENV_FILE = Path(__file__).resolve().parent / ".env"

_client = None
_playlists = None        # (fetched_at, [playlist dict]) -- see _library_playlists()


def _load_env():
    """Same .env as testChatWakeup.py. The real environment always wins."""
    try:
        text = ENV_FILE.read_text()
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def client(interactive=False):
    """The authorised spotipy client, built once and reused.

    Built lazily rather than at import: `import Spotify` happens inside
    ActionManager.builtin_actions(), and an unconfigured Spotify must not stop the assistant
    from starting -- it should just make those actions report why they cannot run.
    """
    global _client
    if _client is not None:
        return _client

    _load_env()
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Spotify is not set up: SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET are missing "
            f"from {ENV_FILE}. See the setup notes at the top of Spotify.py."
        )

    TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    auth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=os.environ.get("SPOTIFY_REDIRECT_URI", REDIRECT_URI),
        scope=SCOPES,
        cache_handler=CacheFileHandler(cache_path=str(TOKEN_CACHE)),
        # There is no browser on this Pi, and mid-conversation there is nobody to answer a
        # prompt. Anything but a cached token is an error the actions report out loud.
        open_browser=False,
    )
    if not interactive and not auth.validate_token(auth.cache_handler.get_cached_token()):
        raise RuntimeError(
            "Spotify is not authorised yet. Run: .venv/bin/python Spotify.py login\n"
            "(a token cached under an older, smaller SCOPES string looks identical to no "
            "token at all -- re-running login is the fix either way)"
        )

    _client = spotipy.Spotify(auth_manager=auth)
    return _client


def _speakable(msg):
    """spotipy puts the whole request URL in exc.msg; only the last line is the reason.

    It matters that this is short and clean: whatever comes back here is handed to Gemini
    and read out loud, so a URL and a newline become spoken noise.
    """
    text = (msg or "").strip()
    return text.rsplit("\n", 1)[-1].strip() or "no reason given"


def _call(name, *args, **kwargs):
    """One API call, with Spotify's errors turned into something speakable.

    Raising is right: Action.invoke turns an exception into {"error": ...}, which reaches
    Gemini as text it can explain -- "nothing is playing" beats a stack trace. But that
    also means a *wrong* explanation here gets asserted out loud as fact, which is what
    happened: every 403 used to have "(playback control needs Premium)" appended, so a
    Premium account was told it was not one. Measured 2026-09-07 with nothing playing:

        PUT /me/player/pause  -> 403 reason='UNKNOWN'
                                 'Player command failed: Restriction violated'

    That is Spotify's "invalid in the current player state" (nothing is playing, or the
    player is already in the state you asked for), and it has nothing to do with the
    account tier. Premium has its own reason, PREMIUM_REQUIRED, so only that says Premium.
    """
    try:
        return getattr(client(), name)(*args, **kwargs)
    except spotipy.SpotifyException as exc:
        reason = getattr(exc, "reason", None)
        detail = _speakable(exc.msg)
        if reason == "NO_ACTIVE_DEVICE" or exc.http_status == 404:
            raise RuntimeError("no active Spotify device") from None
        if exc.http_status == 403:
            if reason == "PREMIUM_REQUIRED" or "premium" in detail.lower():
                raise RuntimeError(
                    f"Spotify says that needs Premium: {detail}"
                ) from None
            if "restriction violated" in detail.lower():
                raise RuntimeError(
                    "Spotify would not accept that right now -- most likely nothing is "
                    "playing, or the player is already in that state"
                ) from None
            raise RuntimeError(f"Spotify refused that: {detail}") from None
        if exc.http_status == 429:
            raise RuntimeError("Spotify is rate limiting this app; try again shortly") from None
        raise RuntimeError(f"Spotify error {exc.http_status}: {detail}") from None
    except spotipy.SpotifyOauthError as exc:
        raise RuntimeError(f"Spotify authorisation failed: {exc}") from None


def _pi_device_id():
    for device in _call("devices").get("devices", []):
        if device["name"] == PI_DEVICE:
            return device["id"]
    return None


# Words a person wraps a request in but that are never part of the name they mean, so
# "my gym playlist" can still reach a playlist called "Gym 2026".
_FILLER = frozenset(
    "a an the my me some please play put on off by from for of and to is it "
    "song songs track tracks album playlist playlists mix music list spotify".split()
)

# Hedges. "play some jazz" and "play Jazz" score identically -- one content word against a
# one-word title -- so no amount of string cleverness separates them. The quantifier is the
# only honest signal, and it says the user described a mood rather than named a thing, so a
# hedged query never commits to a single track no matter how well it scores.
_HEDGES = frozenset("some something anything any whatever".split())

# A score at or below this came only from loose word overlap, with no run of the name said
# in full. Good enough to pick a playlist out of a library the user owns; not good enough to
# commit to one specific track out of everything on Spotify -- there, "some jazz" landing on
# a track called "Jazz" is worse than queueing the page and letting it play.
WEAK_MATCH = 30

# Unicode-aware on purpose: an [a-z0-9] class deletes CJK outright, which normalised the
# library's "100首粤语经典" down to "100" and made it unreachable by the name it has.
_PUNCTUATION = re.compile(r"[^\w ]+", re.UNICODE)
_BRACKETED = re.compile(r"\(.*?\)|\[.*?\]")


def _norm(text):
    """Fold a title down to what a person actually said.

    "Moog City 2 (Remastered)" -> "moog city 2". Accents are stripped because the mic's
    transcript never has them, and bracketed suffixes go because "- Remastered 2011" is
    noise for matching but would otherwise make an exact match impossible.
    """
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = _BRACKETED.sub(" ", text)
    text = text.split(" - ")[0] if " - " in text else text
    return " ".join(_PUNCTUATION.sub(" ", text).split())


def _match_score(query, title, also=()):
    """How well `title` answers `query`. 0 means "no better than Spotify's own order".

    Deliberately a small ladder of whole-word rules rather than an edit distance: the input
    is a speech transcript, so the failure to guard against is a *different* title that
    happens to share letters, not a typo. `also` holds artist names, which only ever add a
    tie-break -- an artist match alone must not beat a title match.
    """
    q, t = _norm(query), _norm(title)
    if not q or not t:
        return 0

    if q == t:
        score = 100
    elif q.startswith(t + " ") or q.endswith(" " + t) or f" {t} " in f" {q} ":
        score = 70                      # "moog city by c418" -- title said in full
    elif t.startswith(q + " "):
        score = 40                      # asked "moog city", this is "moog city 2"
    elif q in t:
        score = 20
    else:
        said = [w for w in q.split() if w not in _FILLER]
        if said and set(said) <= set(t.split()):
            score = WEAK_MATCH      # every real word landed, just not as one phrase
        else:
            return 0

    for name in also:
        n = _norm(name)
        if n and (n == q or f" {n} " in f" {q} "):
            score += 15
            break
    return score


def _best_match(query, items, title=lambda i: i["name"], also=lambda i: ()):
    """(index, score) of the item that best answers `query`, ties going to Spotify's order.

    Returning the score matters as much as the index: a top score of 0 means nothing
    actually matched by name, and the caller should then trust the ranker rather than
    pretend the first row was a hit.
    """
    if not items:
        return None, 0
    scores = [_match_score(query, title(item), also(item)) for item in items]
    best = max(range(len(items)), key=lambda i: (scores[i], -i))
    return best, scores[best]


def _start(shuffle=None, **playback):
    """start_playback, waking the Pi when nothing is playing anywhere.

    Shuffle is set *before* playback starts, because Spotify applies it to the context as it
    loads -- setting it afterwards leaves the first track where it was.
    """
    def go(device_id=None):
        if shuffle is not None:
            _call("shuffle", shuffle, device_id=device_id)
        _call("start_playback", device_id=device_id, **playback)

    try:
        go()
    except RuntimeError as exc:
        if "no active" not in str(exc):
            raise
        # The Pi is a Connect device but an idle one, and idle is not active. Wake it.
        device_id = _pi_device_id()
        if device_id is None:
            raise RuntimeError(
                f"no Spotify device is available, and {PI_DEVICE} is not in the list "
                "(is the librespot service running?)"
            ) from None
        _call("transfer_playback", device_id=device_id, force_play=False)
        go(device_id)


def _library_playlists(refresh=False):
    """Every playlist in the account's library, paged and briefly cached.

    Cached because a voice turn cannot afford four HTTP round trips before the music starts,
    and the library changes far more slowly than PLAYLIST_TTL. The `if entry` filter is not
    defensive noise: /me/playlists genuinely returns null rows for playlists that have been
    deleted out from under the index.
    """
    global _playlists
    if not refresh and _playlists and time.monotonic() - _playlists[0] < PLAYLIST_TTL:
        return _playlists[1]

    found, offset = [], 0
    while True:
        page = _call("current_user_playlists", limit=50, offset=offset)
        found.extend(entry for entry in page.get("items", []) if entry)
        if not page.get("next") or offset >= 950:
            break
        offset += 50

    _playlists = (time.monotonic(), found)
    return found


def _find_playlist(name):
    """The library playlist called `name`, or a RuntimeError that says what is there."""
    for refresh in (False, True):
        playlists = _library_playlists(refresh=refresh)
        index, score = _best_match(name, playlists)
        if score:
            return playlists[index]
        # A miss on the cache is the one case worth paying for a re-fetch: a playlist made
        # since the last call is exactly the one someone is most likely to ask for.

    known = [entry["name"] for entry in _library_playlists()]
    if not known:
        raise RuntimeError("there are no playlists in your Spotify library")
    listed = ", ".join(known[:8]) + (", ..." if len(known) > 8 else "")
    raise RuntimeError(f"no playlist matches {name!r}; the library has: {listed}")


def _describe(playback):
    if not playback or not playback.get("item"):
        return "nothing is playing"
    item = playback["item"]
    artists = ", ".join(a["name"] for a in item.get("artists", []))
    where = (playback.get("device") or {}).get("name", "unknown device")
    state = "playing" if playback.get("is_playing") else "paused"
    return f"{item['name']} by {artists} -- {state} on {where}"


# --- the actions ----------------------------------------------------------


def spotify_play(query: str = ""):
    """Play, waking the Pi itself if nothing is playing anywhere.

    A named song and a vague mood are two different requests wearing the same argument, and
    the top score (plus the hedge check) tells them apart. "Moog City" matches a title, so
    exactly that track is played; "some jazz" is a mood, so Spotify's own ranking is trusted
    and the whole page is queued -- otherwise "play some jazz" would stop dead after one
    track, on whatever single song happened to be called "Jazz".
    """
    track = None
    uris = None
    if query:
        found = _call("search", q=query, type="track", limit=SEARCH_LIMIT)
        items = found.get("tracks", {}).get("items", [])
        if not items:
            return f"nothing on Spotify matches {query!r}"
        index, score = _best_match(
            query, items, also=lambda item: [a["name"] for a in item.get("artists", [])]
        )
        hedged = bool(_HEDGES & set(_norm(query).split()))
        if score > WEAK_MATCH and not hedged:
            track = items[index]
            uris = [track["uri"]]
        else:
            uris = [item["uri"] for item in items]

    _start(uris=uris)

    if track:
        return f"playing {track['name']} by {track['artists'][0]['name']}"
    if uris:
        return f"playing what Spotify has for {query!r}"
    return "playing"


def spotify_playlist(name: str, shuffle: bool = True):
    """Play a playlist from the account's own library, by name."""
    playlist = _find_playlist(name)
    _start(context_uri=playlist["uri"], shuffle=shuffle)
    total = (playlist.get("tracks") or {}).get("total")
    count = f", {total} tracks" if total else ""
    return f"playing {playlist['name']}{count}{' on shuffle' if shuffle else ''}"


def spotify_playlists():
    """The names of the playlists in the library, for when the user asks what is there."""
    names = [entry["name"] for entry in _library_playlists()]
    if not names:
        return "there are no playlists in your Spotify library"
    if len(names) > 20:
        return f"{len(names)} playlists, including: " + ", ".join(names[:20])
    return ", ".join(names)


def spotify_pause():
    _call("pause_playback")
    return "paused"


def spotify_next():
    _call("next_track")
    return "skipped"


def spotify_previous():
    _call("previous_track")
    return "went back"


def spotify_status():
    return _describe(_call("current_playback"))


def spotify_volume(level: int):
    if not 0 <= level <= 100:
        raise ValueError("volume must be between 0 and 100")
    _call("volume", level)
    return f"volume {level}%"


def spotify_shuffle(mode: Literal["on", "off"]):
    _call("shuffle", mode == "on")
    return f"shuffle {mode}"


def spotify_devices():
    found = _call("devices").get("devices", [])
    if not found:
        return "no Spotify devices are available"
    return ", ".join(
        f"{d['name']}{' (active)' if d['is_active'] else ''}" for d in found
    )


def actions():
    """Registered by ActionManager.builtin_actions() when this module imports cleanly."""
    return [
        Action(
            "Start or resume Spotify playback, optionally searching for something first. "
            "Call this for 'play some jazz', 'put on <artist>', or a bare 'play'.",
            spotify_play,
            describe={
                "query": "what to search for, e.g. an artist, song or genre; "
                "leave empty to resume whatever was paused"
            },
        ),
        Action(
            "Play a playlist from the user's own Spotify library. Call this whenever a "
            "request names a playlist, or says 'my <something> playlist' -- do not use "
            "the general play action for that, it only searches songs.",
            spotify_playlist,
            describe={
                "name": "the playlist's name as the user said it; it is matched loosely "
                "against the library, so an approximate name is fine",
                "shuffle": "true unless the user asks for it in order",
            },
        ),
        Action(
            "List the playlists in the user's Spotify library. Call this when asked what "
            "playlists exist, or after a playlist name was not found.",
            spotify_playlists,
        ),
        Action("Pause Spotify playback.", spotify_pause),
        Action("Skip to the next track on Spotify.", spotify_next),
        Action("Go back to the previous track on Spotify.", spotify_previous),
        Action(
            "Find out what is playing on Spotify right now -- track, artist and whether it "
            "is paused. Call this when asked what song this is.",
            spotify_status,
        ),
        Action(
            "Set the Spotify player's volume. This is the music volume, not how loudly you "
            "speak.",
            spotify_volume,
            describe={"level": "0 to 100 percent"},
        ),
        Action("Turn Spotify shuffle on or off.", spotify_shuffle),
        Action(
            "List the Spotify players this account can reach and which one is active. Call "
            "this when asked where the music is playing, or to explain why playback will "
            "not start.",
            spotify_devices,
        ),
    ]


# --- setup and diagnostics ------------------------------------------------


def login_via_server(port=8888, timeout=300):
    """Catch the redirect on a real listener here, reached through an SSH tunnel.

    spotipy's start_local_http_server binds to 127.0.0.1, which is exactly what
    `ssh -L 8888:127.0.0.1:8888` forwards to -- so the browser on your laptop reaches a socket
    on the Pi and the flow completes with nothing to copy by hand. spotipy's own local-server
    path is not reused because it calls webbrowser.open() first, which is meaningless here.
    """
    auth = client(interactive=True).auth_manager
    try:
        server = start_local_http_server(port)
    except OSError as exc:
        raise SystemExit(
            f"Could not listen on 127.0.0.1:{port}: {exc}\n"
            "Something else is using it, or a previous login is still running."
        ) from None

    print(f"Listening on 127.0.0.1:{port} for the redirect.\n")
    print("On your laptop, make sure the tunnel is up:\n")
    print(f"  ssh -L {port}:127.0.0.1:{port} johnny@<this-pi>\n")
    print("Then open this and approve:\n")
    print(f"  {auth.get_authorize_url()}\n")
    print(f"Waiting up to {timeout // 60} minutes... Ctrl-C to give up.")

    server.timeout = timeout
    server.handle_request()          # blocks until the browser arrives
    if server.error is not None:
        raise SystemExit(f"The redirect reported an error: {server.error}")
    if not server.auth_code:
        raise SystemExit(
            "Nothing arrived. Either the tunnel is not up, or the browser never reached "
            f"127.0.0.1:{port}. Fall back to: Spotify.py login \"<redirected url>\""
        )

    auth.get_access_token(server.auth_code, as_dict=False)
    print(f"\nAuthorised. Token cached in {TOKEN_CACHE}")


def login(response=None):
    """One-time authorisation, headless-friendly.

    spotipy would normally spin up a local web server to catch the redirect, which is no use
    when you are authorising from a laptop against a Pi over SSH. With open_browser=False it
    prints the URL and takes the redirected URL back by hand instead.

    `response` lets the two halves happen in either order:

        python Spotify.py login              # prints the URL, waits for the paste
        python Spotify.py login "<url>"      # redeem a URL you already have

    which matters because the redirect deliberately goes nowhere -- the browser tab hangs, and
    it is easy to close it before coming back to the terminal. The authorisation is already
    done at that point; the code is sitting in the address bar.
    """
    auth = client(interactive=True).auth_manager

    if response is None:
        print(f"The redirect URI registered in your app must be: {auth.redirect_uri}\n")
        print("Open this in any browser (your laptop is fine) and approve:\n")
        print(f"  {auth.get_authorize_url()}\n")
        print(
            "The tab will then hang on a 127.0.0.1 page that never loads -- expected, since\n"
            "nothing is listening there. The code is already in the address bar. Copy the\n"
            "whole URL and paste it here (or re-run: Spotify.py login \"<url>\").\n"
        )
        response = input("Redirected URL: ").strip()

    code = auth.parse_response_code(response.strip())
    if code == response.strip():
        raise SystemExit(
            "That does not look like a redirect URL -- it needs the ?code=... part, e.g.\n"
            f"  {auth.redirect_uri}?code=AQD9x..."
        )
    try:
        auth.get_access_token(code, as_dict=False)
    except Exception as exc:
        # Codes are single-use and expire in minutes, which is the usual cause.
        raise SystemExit(
            f"Could not redeem that code: {exc}\n"
            "Authorisation codes are single use and expire quickly -- run login again."
        ) from None
    print(f"\nAuthorised. Token cached in {TOKEN_CACHE}")


def selftest():
    """Check the chooser against recorded search results. No API key, no network.

    The fixtures are real limit=10 pages captured from this account, kept because the point
    of _match_score is repeatability: if a rung is retuned, these say what it cost.
    """
    def tracks(pairs):
        return [{"name": n, "uri": f"spotify:track:{i}", "artists": [{"name": a}]}
                for i, (n, a) in enumerate(pairs)]

    moog = tracks([("Moog City", "C418"), ("Moog City 2", "C418"), ("Aria Math", "C418"),
                   ("Mice on Venus", "C418"), ("Sweden", "C418"),
                   ("Moog City - Slowed + Rain", "ambientique"),
                   ("moog city (ambient)", "exhibit"), ("MOOGCiTY", "AKIBA")])
    sweden = tracks([("Sweden", "C418"), ("Mice on Venus", "C418"), ("Aria Math", "C418"),
                     ("Sweden - C418 - Remix", "Technizite"), ("Minecraft", "C418"),
                     ("Sweden", "AngryParkRanger"), ("Sweden", "sk4le")])
    queen = tracks([("Bohemian Rhapsody", "Queen"), ("Bohemian Rhapsody", "Queen"),
                    ("Bohemian Rhapsody - Remastered", "Queen"),
                    ("Bohemian Rhapsody", "Panic! At The Disco"),
                    ("Bohemian Rhapsody", "Pentatonix")])
    jazz = tracks([("Blue in Green", "Miles Davis"), ("Jazz", "Ella"),
                   ("Take Five", "Brubeck")])
    library = [{"name": n} for n in
               ["Study Beats", "Gym 2026", "chill lofi", "Liked from Radio", "Study"]]
    artists = lambda item: [a["name"] for a in item["artists"]]

    named = [
        # the reported bug: limit=1 answered "Aria Math" for this query
        ("Moog City", moog, "Moog City", "C418"),
        ("play Moog City", moog, "Moog City", "C418"),
        ("Moog City 2", moog, "Moog City 2", "C418"),
        ("Sweden", sweden, "Sweden", "C418"),
        ("sweden by c418", sweden, "Sweden", "C418"),
        ("Bohemian Rhapsody", queen, "Bohemian Rhapsody", "Queen"),
        ("Bohemian Rhapsody by Panic at the Disco", queen,
         "Bohemian Rhapsody", "Panic! At The Disco"),
        ("Pentatonix Bohemian Rhapsody", queen, "Bohemian Rhapsody", "Pentatonix"),
    ]
    moods = [("some jazz", jazz), ("something upbeat", moog), ("play some music", jazz)]
    playlists = [("study beats", "Study Beats"), ("my gym playlist", "Gym 2026"),
                 ("the chill lofi one", "chill lofi"), ("Study", "Study"),
                 ("gym", "Gym 2026"), ("play my Gym 2026 playlist", "Gym 2026")]

    failures = 0
    for query, items, title, artist in named:
        index, score = _best_match(query, items, also=artists)
        got = items[index]
        good = (got["name"], got["artists"][0]["name"]) == (title, artist)
        good &= score > WEAK_MATCH and not (_HEDGES & set(_norm(query).split()))
        failures += not good
        print(f"  {'ok  ' if good else 'FAIL'} track {query!r:42} -> "
              f"{got['name']!r} by {got['artists'][0]['name']} ({score})")

    for query, items in moods:
        index, score = _best_match(query, items, also=artists)
        hedged = bool(_HEDGES & set(_norm(query).split()))
        good = hedged or score <= WEAK_MATCH        # must queue the page, not one track
        failures += not good
        print(f"  {'ok  ' if good else 'FAIL'} mood  {query!r:42} -> queue {len(items)} "
              f"(score {score}, hedged {hedged})")

    for query, want in playlists:
        index, score = _best_match(query, library)
        good = score > 0 and library[index]["name"] == want
        failures += not good
        print(f"  {'ok  ' if good else 'FAIL'} list  {query!r:42} -> "
              f"{library[index]['name']!r} ({score})")

    index, score = _best_match("nothing like this exists", library)
    failures += score != 0
    print(f"  {'ok  ' if score == 0 else 'FAIL'} list  {'a name nobody has':44} -> "
          f"no match ({score})")

    print(f"\n{'all pass' if not failures else str(failures) + ' FAILED'}")
    return failures


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(1 if selftest() else 0)

    if len(sys.argv) > 1 and sys.argv[1] == "login":
        argument = sys.argv[2] if len(sys.argv) > 2 else None
        if argument == "--server":
            login_via_server()
        else:
            login(argument)
        raise SystemExit(0)

    try:
        client()
        print(f"authorised, token cache: {TOKEN_CACHE}")
        print(f"devices: {spotify_devices()}")
        print(f"status:  {spotify_status()}")
        try:
            playlists = _library_playlists()
            print(f"library: {len(playlists)} playlists -- {spotify_playlists()}")
        except RuntimeError as exc:
            print(f"library: unavailable -- {exc}")
    except RuntimeError as exc:
        print(f"NOT READY -- {exc}")

    print(f"\nactions: {len(actions())}")
    for action in actions():
        print(f"  {action.summary()}")
