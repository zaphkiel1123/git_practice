#!/usr/bin/env python3
"""
Serve the tick-based candlestick replay viewer for an ES .data tick file.

    python3 view_ticks.py ../20250213.data
    python3 view_ticks.py ../20250213.data --interval 5      # 5 s candles (1, 5, 15 or 60)
    python3 view_ticks.py ../20250213.data --bar 500         # 500-tick candles
    python3 view_ticks.py ../20250213.data --port 8765 --host 0.0.0.0

Standard library only. The viewer itself (tick_viewer.html) decodes the binary
file in the browser, so there is no preprocessing step. Timestamps in the file
are New York wall-clock time and are displayed as-is.

If you'd rather not run a server, open tick_viewer.html directly in a browser
and pick / drop the .data file.
"""

import argparse
import os
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "tick_viewer.html")


def make_handler(data_path, on_post=None):
    """on_post(route, body) -> HTTP status; enables POST (used by record_replay.py)."""
    data_name = os.path.basename(data_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):      # keep the terminal quiet
            pass

        def _send_file(self, path, ctype, extra=None):
            try:
                size = os.path.getsize(path)
                f = open(path, "rb")
            except OSError as e:
                self.send_error(404, str(e))
                return
            with f:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(size))
                self.send_header("Cache-Control", "no-store")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                try:
                    while True:
                        chunk = f.read(1 << 20)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            route = self.path.split("?", 1)[0]
            if route in ("/", "/index.html", "/tick_viewer.html"):
                self._send_file(HTML, "text/html; charset=utf-8")
            elif route == "/data":
                self._send_file(data_path, "application/octet-stream",
                                {"X-File-Name": data_name})
            else:
                self.send_error(404)

        def do_POST(self):
            if on_post is None:
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n > 0 else b""
            code = on_post(self.path.split("?", 1)[0], body)
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default=None,
                    help="path to a YYYYMMDD.data file (default: ../20250213.data)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1; use 0.0.0.0 to reach it from another machine)")
    ap.add_argument("--port", type=int, default=8765, help="port (0 = pick a free one)")
    ap.add_argument("--no-browser", action="store_true", help="don't try to open a browser")
    ap.add_argument("--start", default=None, metavar="HH:MM[:SS]",
                    help="ET time to start the replay at (default 09:30:00)")
    ap.add_argument("--bar", type=int, default=None, help="initial bar size in ticks (tick bars)")
    ap.add_argument("--interval", type=int, choices=(1, 5, 15, 60), default=None,
                    help="use time bars of this many seconds (1, 5, 15, 60) instead of tick bars")
    ap.add_argument("--no-footprint", action="store_true",
                    help="start with plain candles (toggle with the Footprint button or F)")
    args = ap.parse_args()

    data = args.data or os.path.join(HERE, "..", "20250213.data")
    data = os.path.abspath(data)
    if not os.path.isfile(data):
        sys.exit(f"data file not found: {data}")
    size = os.path.getsize(data)
    if size % 57:
        print(f"warning: {size} bytes is not a multiple of the 57-byte record size", file=sys.stderr)

    try:
        srv = ThreadingHTTPServer((args.host, args.port), make_handler(data))
    except OSError as e:
        sys.exit(f"cannot bind {args.host}:{args.port}: {e}")
    port = srv.server_address[1]

    q = []
    if args.start:
        q.append(f"t={args.start}")
    if args.interval:
        q.append(f"unit=time&ivl={args.interval}")
    elif args.bar:
        q.append(f"n={args.bar}")
    if args.no_footprint:
        q.append("fp=0")
    suffix = ("?" + "&".join(q)) if q else ""
    shown_host = socket.getfqdn() if args.host in ("0.0.0.0", "") else args.host
    url = f"http://{shown_host}:{port}/{suffix}"

    print(f"Serving {data} ({size // 57:,} records)")
    print(f"Open: {url}")
    if args.host in ("127.0.0.1", "localhost"):
        print(f"(remote machine? tunnel it:  ssh -L {port}:localhost:{port} {socket.gethostname()})")
    print("Ctrl-C to stop.")

    if not args.no_browser and (os.environ.get("DISPLAY") or sys.platform != "linux"):
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
