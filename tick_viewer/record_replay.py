#!/usr/bin/env python3
"""
Record a replay of an ES .data tick file to an MP4 or WebM video, headless (Linux, Windows, macOS).

    python3 record_replay.py ../20250213.data --at 09:30:00             # 10 s at 1x from 09:30:00 ET
    python3 record_replay.py ../20250213.data --biggest                 # around the largest print
    python3 record_replay.py ../20250213.data --at 13:33:00 --seconds 20 --speed 10 --interval 5
    py record_replay.py ..\\20250213.data --biggest                      (Windows)

Standard library only. The script serves tick_viewer.html through view_ticks.py, opens it in a
headless browser, and the page records its own chart, Time & Sales tape and status bar with
MediaRecorder and uploads the video back here. See README.md.

Format: --format auto (default) is mp4 on Windows and webm elsewhere; -o name.mp4 also asks for mp4.
MP4 is recorded natively by Chrome / Edge 126+ (preferred browsers when mp4 is wanted). Otherwise
the WebM is converted with ffmpeg (--ffmpeg, PATH, ./bin/, or pip's imageio-ffmpeg); without
ffmpeg the WebM is kept and the script says so.

The recording runs in wall-clock time: --seconds 10 --speed 1 shows 10 s of market time,
--seconds 10 --speed 10 shows 100 s. Playback is clock driven, so if the browser drops frames the
video is choppier but still covers the right stretch of market time.
"""

import argparse
import datetime
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from view_ticks import make_handler  # noqa: E402

REC_SIZE = 57
NET_EPOCH_TO_UNIX_MS = 62135596800000


def _win_paths(rel):
    roots = [os.environ.get(k) for k in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")]
    return [os.path.join(r, rel) for r in roots if r]


# name -> (PATH names, Windows install paths, macOS app paths)
BROWSERS = {
    "firefox": (["firefox"], _win_paths(r"Mozilla Firefox\firefox.exe"),
                ["/Applications/Firefox.app/Contents/MacOS/firefox"]),
    "chrome": (["google-chrome", "google-chrome-stable", "chrome"], _win_paths(r"Google\Chrome\Application\chrome.exe"),
               ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]),
    "edge": (["microsoft-edge", "msedge"], _win_paths(r"Microsoft\Edge\Application\msedge.exe"),
             ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]),
    "chromium": (["chromium", "chromium-browser"], _win_paths(r"Chromium\Application\chrome.exe"),
                 ["/Applications/Chromium.app/Contents/MacOS/Chromium"]),
}

FIREFOX_PREFS = """\
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.aboutwelcome.enabled", false);
user_pref("datareporting.policy.dataSubmissionEnabled", false);
user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);
user_pref("app.update.enabled", false);
user_pref("browser.startup.homepage_override.mstone", "ignore");
"""


def find_browser(choice, want_mp4=False):
    """-> (kind, executable). choice: None, a browser name from BROWSERS, or a path.
    With no choice, Chromium-family browsers go first when MP4 is wanted (Firefox can only record WebM)."""
    if choice and choice not in BROWSERS:
        if not os.path.isfile(choice):
            sys.exit(f"browser not found: {choice}")
        return ("firefox" if "firefox" in os.path.basename(choice).lower() else "chromium"), choice
    order = ["edge", "chrome", "chromium", "firefox"] if want_mp4 else list(BROWSERS)
    for name in ([choice] if choice else order):
        on_path, win, mac = BROWSERS[name]
        for exe in on_path:
            found = shutil.which(exe)
            if found:
                return ("firefox" if name == "firefox" else "chromium"), found
        for p in (win if os.name == "nt" else mac if sys.platform == "darwin" else []):
            if os.path.isfile(p):
                return ("firefox" if name == "firefox" else "chromium"), p
    sys.exit("no Firefox / Chrome / Edge / Chromium found; pass --browser <path to the executable>")


def browser_cmd(kind, exe, url, width, height, profile):
    if kind == "firefox":
        with open(os.path.join(profile, "user.js"), "w") as f:
            f.write(FIREFOX_PREFS)
        cmd = [exe, "--headless", "-no-remote", "-new-instance", "-profile", profile,
               f"--window-size={width},{height}"]
        if os.name == "nt":
            cmd.append("-wait-for-browser")     # keep the launcher alive so taskkill /T reaches the browser
        cmd.append(url)
    else:
        cmd = [exe, "--headless=new", f"--user-data-dir={profile}", f"--window-size={width},{height}",
               "--no-first-run", "--no-default-browser-check", "--mute-audio",
               "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
               "--disable-backgrounding-occluded-windows", url]
    # Some Linux browser wrappers insist on an X display even in headless mode.
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a", "-s", f"-screen 0 {max(width, 1920)}x{max(height, 1080)}x24"] + cmd
    return cmd


def start_browser(cmd, log):
    if os.name == "nt":
        return subprocess.Popen(cmd, stdout=log, stderr=log, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    return subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)


def stop_browser(proc):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10)
            time.sleep(1)                       # let the rest of the process group go too
        except subprocess.TimeoutExpired:
            pass


def _vint(b, i, keep_marker):
    first = b[i]
    n = 1
    while n <= 8 and not first & (0x80 >> (n - 1)):
        n += 1
    if n > 8:
        raise ValueError("bad EBML vint")
    v = first if keep_marker else first & ((0x80 >> (n - 1)) - 1)
    for k in range(1, n):
        v = (v << 8) | b[i + k]
    unknown = not keep_marker and v == (1 << (7 * n)) - 1
    return v, n, unknown


def fix_webm_duration(path, frame_ms=33):
    """Browsers' MediaRecorder leaves the WebM Duration at 0 (players then guess the length and seek
    badly). Set it in place from the last frame's timestamp. Returns seconds, or None if not patched."""
    with open(path, "rb") as f:
        b = bytearray(f.read())
    containers = {0x18538067, 0x1F43B675, 0x1549A966, 0xA0}   # Segment, Cluster, Info, BlockGroup
    scale_ns, dur_at, dur_len, cluster_tc, last = 1000000, None, 0, 0, None
    i = 0
    try:
        while i < len(b):
            eid, n, _ = _vint(b, i, True)
            size, m, unknown = _vint(b, i + n, False)
            data = i + n + m
            if eid in containers:
                i = data
                continue
            if unknown:
                break
            if eid == 0x2AD7B1:                                  # TimecodeScale
                scale_ns = int.from_bytes(b[data:data + size], "big")
            elif eid == 0x4489:                                  # Duration
                dur_at, dur_len = data, size
            elif eid == 0xE7:                                    # Cluster Timecode
                cluster_tc = int.from_bytes(b[data:data + size], "big")
            elif eid in (0xA3, 0xA1):                            # SimpleBlock / Block
                _, tn, _ = _vint(b, data, False)
                rel = struct.unpack_from(">h", b, data + tn)[0]
                t = cluster_tc + rel
                last = t if last is None or t > last else last
            i = data + size
    except (ValueError, IndexError, struct.error):
        pass
    if dur_at is None or dur_len not in (4, 8) or last is None:
        return None
    ticks = last + frame_ms * 1e6 / scale_ns
    b[dur_at:dur_at + dur_len] = struct.pack(">d" if dur_len == 8 else ">f", ticks)
    with open(path, "wb") as f:
        f.write(b)
    return ticks * scale_ns / 1e9


def sniff_container(path):
    with open(path, "rb") as f:
        head = f.read(12)
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if head[4:8] == b"ftyp":
        return "mp4"
    return None


def find_ffmpeg(choice):
    if choice:
        if not os.path.isfile(choice):
            sys.exit(f"ffmpeg not found: {choice}")
        return choice
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for p in (shutil.which("ffmpeg"), os.path.join(HERE, "bin", exe)):
        if p and os.path.isfile(p):
            return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def run_ffmpeg(ffmpeg, src, dst, transcode):
    """WebM -> H.264 MP4 (transcode) or MP4 -> MP4 remux with the index up front. -> error text or None."""
    base = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", src]
    tail = ["-movflags", "+faststart", "-an", dst]
    if not transcode:
        attempts = [["-c", "copy"]]
    else:
        common = ["-pix_fmt", "yuv420p"]
        attempts = [["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"] + common,
                    ["-c:v", "libopenh264", "-b:v", "6M"] + common]
        if os.name == "nt":
            attempts.append(["-c:v", "h264_mf", "-b:v", "6M"] + common)
    err = None
    for a in attempts:
        r = subprocess.run(base + a + tail, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode == 0 and os.path.isfile(dst) and os.path.getsize(dst) > 0:
            return None
        err = r.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [f"exit {r.returncode}"]
        err = err[0]
    return err


def ms_to_dt(ms):
    return datetime.datetime(1970, 1, 1) + datetime.timedelta(milliseconds=ms)


def find_biggest(path):
    """-> (size, ms, price) of the largest single print; ms is ET wall clock as if UTC."""
    with open(path, "rb") as f:
        buf = f.read()
    best = (-1, 0, 0.0)
    for o in range(0, len(buf) - REC_SIZE + 1, REC_SIZE):
        q = struct.unpack_from("<I", buf, o + 32)[0]
        if q > best[0]:
            t, p = struct.unpack_from("<qd", buf, o)
            best = (q, t // 10000 - NET_EPOCH_TO_UNIX_MS, p)
    return best


def parse_hms(s):
    parts = s.strip().split(":")
    if len(parts) not in (2, 3) or not all(x.isdigit() for x in parts):
        sys.exit(f"bad time {s!r}, expected HH:MM[:SS]")
    h, m, sec = (int(x) for x in parts + ["0"] * (3 - len(parts)))
    if h > 23 or m > 59 or sec > 59:
        sys.exit(f"bad time {s!r}")
    return f"{h:02d}:{m:02d}:{sec:02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default=os.path.join(HERE, "..", "20250213.data"),
                    help="path to a YYYYMMDD.data file (default: ../20250213.data)")
    when = ap.add_mutually_exclusive_group()
    when.add_argument("--at", metavar="HH:MM[:SS]", help="ET time to start at (default 09:30:00)")
    when.add_argument("--biggest", action="store_true",
                      help="start shortly before the largest single print in the file")
    ap.add_argument("--lead", type=float, default=None, metavar="SEC",
                    help="with --biggest: market seconds before the print (default 30%% of the clip)")
    ap.add_argument("--seconds", type=float, default=10, help="video length in seconds (default 10)")
    ap.add_argument("--speed", type=float, default=1, help="real-time multiplier, or ticks/s with --rate (default 1)")
    ap.add_argument("--rate", action="store_true", help="--speed is ticks per second instead of a multiplier")
    ap.add_argument("--bar", type=int, default=None, help="bar size in ticks (viewer default 200)")
    ap.add_argument("--interval", type=int, choices=(1, 5, 15, 60), default=None, help="time bars of N seconds")
    ap.add_argument("--vis", type=int, default=None, help="bars on screen")
    ap.add_argument("--no-footprint", action="store_true", help="plain candles")
    ap.add_argument("--no-tape", action="store_true", help="leave out the Time & Sales tape")
    ap.add_argument("--size", default="1600x900", help="video WxH, even numbers (default 1600x900)")
    ap.add_argument("--browser", default=None,
                    help="firefox | chrome | edge | chromium | path to the executable (default: first found)")
    ap.add_argument("--format", choices=("auto", "mp4", "webm"), default="auto",
                    help="auto = from -o's extension, else mp4 on Windows and webm elsewhere")
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg executable for WebM -> MP4 (default: PATH, ./bin, imageio-ffmpeg)")
    ap.add_argument("-o", "--out", default=None,
                    help="output .mp4 / .webm (default recordings/replay_<file>_<time>_<len>s.<format>)")
    ap.add_argument("--timeout", type=float, default=None, help="give up after this many seconds (default length + 180)")
    args = ap.parse_args()

    data = os.path.abspath(args.data)
    if not os.path.isfile(data):
        sys.exit(f"data file not found: {data}")
    if args.ffmpeg and not os.path.isfile(args.ffmpeg):
        sys.exit(f"ffmpeg not found: {args.ffmpeg}")
    try:
        width, height = (int(x) for x in args.size.lower().split("x"))
    except ValueError:
        sys.exit(f"bad --size {args.size!r}, expected WxH")

    if args.biggest:
        q, ms, price = find_biggest(data)
        lead = args.lead if args.lead is not None else (
            max(1.0, 0.3 * args.seconds * args.speed) if not args.rate else 1.0)
        start_ms = ms - int(lead * 1000)
        start = ms_to_dt(start_ms).strftime("%H:%M:%S")
        print(f"Largest print: {q} contracts @ {price:.2f} at {ms_to_dt(ms):%Y-%m-%d %H:%M:%S}."
              f"{ms % 1000:03d} ET -> starting at {start} ET")
    else:
        start = parse_hms(args.at or "09:30:00")

    out_ext = os.path.splitext(args.out)[1].lower().lstrip(".") if args.out else ""
    fmt = args.format
    if fmt == "auto":
        fmt = out_ext if out_ext in ("mp4", "webm") else ("mp4" if os.name == "nt" else "webm")
    elif out_ext in ("mp4", "webm") and out_ext != fmt:
        sys.exit(f"--format {fmt} does not match -o {args.out}")
    stem = os.path.splitext(os.path.basename(data))[0]
    out = os.path.abspath(args.out or os.path.join(
        HERE, "recordings", f"replay_{stem}_{start.replace(':', '')}_{args.seconds:g}s.{fmt}"))
    if out_ext not in ("mp4", "webm"):
        out = os.path.splitext(out)[0] + "." + fmt
    timeout = args.timeout or args.seconds + 180

    q = [f"t={start}", f"rec={args.seconds:g}", f"size={width}x{height}",
         f"mode={'rate' if args.rate else 'rt'}", f"speed={args.speed:g}"]
    if args.interval:
        q.append(f"unit=time&ivl={args.interval}")
    elif args.bar:
        q.append(f"n={args.bar}")
    if args.vis:
        q.append(f"vis={args.vis}")
    if args.no_footprint:
        q.append("fp=0")
    if args.no_tape:
        q.append("tape=0")

    kind, exe = find_browser(args.browser, want_mp4=(fmt == "mp4"))
    print(f"Browser: {exe}")
    print(f"Format:  {fmt}", flush=True)
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="tick_rec_")
    try:
        raw = os.path.join(tmp, "upload.bin")
        page_fmt = "mp4" if fmt == "mp4" and kind != "firefox" else "webm"
        state = record(data, q + [f"fmt={page_fmt}"], kind, exe, width, height, timeout, tmp, raw)
        if state["error"] and page_fmt == "mp4" and state["mp4"]:
            print(f"Browser MP4 recording failed ({state['error']}); retrying as WebM", flush=True)
            state = record(data, q + ["fmt=webm"], kind, exe, width, height, timeout, tmp, raw)
        if state["error"]:
            print(f"FAILED: {state['error']}", file=sys.stderr)
            if state["log"]:
                print("--- browser output (last 20 lines) ---\n" + "\n".join(state["log"]), file=sys.stderr)
            sys.exit(1)
        saved, note = finish(raw, out, fmt, args.ffmpeg)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"Saved {saved} ({os.path.getsize(saved) / 1e6:.1f} MB, {note}, {time.time() - t0:.0f} s total)")


def record(data, query, kind, exe, width, height, timeout, tmp, raw):
    """One browser run. -> {'error', 'mp4' (recorder started in MP4), 'log' (browser output tail)}."""
    done = threading.Event()
    state = {"error": None, "bytes": 0, "status": False, "mp4": False, "log": []}

    def on_post(route, body):
        if route == "/recstatus":
            msg = body.decode("utf-8", "replace")
            state["status"] = True
            print(f"  viewer: {msg}", flush=True)
            if msg.startswith("recording") and "video/mp4" in msg:
                state["mp4"] = True
            if msg.startswith("error"):
                state["error"] = msg
                done.set()
            elif msg == "done":
                done.set()
            return 200
        if route == "/upload":
            with open(raw, "wb") as f:
                f.write(body)
            state["bytes"] = len(body)
            return 200
        return 404

    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(data, on_post))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/?{'&'.join(query)}"
    profile = tempfile.mkdtemp(prefix="profile_", dir=tmp)
    log_path = os.path.join(tmp, "browser.log")
    print(f"URL:     {url}", flush=True)

    t0 = time.time()
    exited_at = None
    with open(log_path, "wb") as log:
        proc = start_browser(browser_cmd(kind, exe, url, width, height, profile), log)
        try:
            while not done.wait(1):
                now = time.time()
                if now - t0 > timeout:
                    state["error"] = f"timed out after {timeout:g} s"
                    break
                if proc.poll() is not None and exited_at is None:
                    exited_at = now
                if exited_at and not state["status"] and now - exited_at > 30:
                    state["error"] = f"browser exited (code {proc.returncode}) before the page reported back"
                    break
        except KeyboardInterrupt:
            state["error"] = "interrupted"
        finally:
            stop_browser(proc)
            srv.shutdown()
            srv.server_close()
    if not state["error"] and not state["bytes"]:
        state["error"] = "no video received"
    if state["error"]:
        with open(log_path, "rb") as f:
            state["log"] = f.read().decode("utf-8", "replace").splitlines()[-20:]
    return state


def finish(raw, out, fmt, ffmpeg_choice):
    """Move the uploaded video to out, converting WebM -> MP4 when needed. -> (saved path, note)."""
    got = sniff_container(raw)
    if got is None:
        sys.exit("FAILED: the uploaded file is neither WebM nor MP4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    note = ""
    if got == "webm":
        dur = fix_webm_duration(raw)
        note = f"{dur:.2f} s" if dur else "duration header not patched"
    ffmpeg = find_ffmpeg(ffmpeg_choice) if fmt == "mp4" else None
    if fmt == "mp4" and ffmpeg:
        # Browser MP4s are fragmented; a remux puts a normal index up front so every player can seek.
        err = run_ffmpeg(ffmpeg, raw, out, transcode=(got == "webm"))
        if err is None:
            how = "H.264 MP4 converted from WebM by ffmpeg" if got == "webm" else "MP4 recorded by the browser"
            return out, how + (f", {note}" if note else "")
        print(f"warning: ffmpeg failed ({err})", file=sys.stderr)
    if fmt == "mp4" and got == "webm":
        out = os.path.splitext(out)[0] + ".webm"
        print("warning: MP4 needs Chrome / Edge 126+ or ffmpeg (see README.md); kept the WebM instead",
              file=sys.stderr)
    elif got != fmt:
        out = os.path.splitext(out)[0] + "." + got
    shutil.move(raw, out)
    return out, (f"{got.upper()} recorded by the browser" + (f", {note}" if note else ""))


if __name__ == "__main__":
    main()
