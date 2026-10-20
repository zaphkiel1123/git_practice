# Tick viewer: recording a replay

`record_replay.py` turns a stretch of an ES `.data` tick file into a video. It plays the replay in a
browser running in the background (headless, nothing appears on screen) and saves what the chart
shows: candles with footprint, the Time & Sales tape and the status bar.

| File | What it is |
|------|------------|
| `record_replay.py` | The recorder. Run this. |
| `tick_viewer.html` | The chart / replay viewer. The recorder drives it; you can also use it interactively. |
| `view_ticks.py` | Small local web server for the viewer (used by the recorder, or on its own). |
| `recordings/` | Default output folder. |

## What you need

Both systems need **Python 3.8 or newer**. Nothing has to be installed with `pip`.

| | Windows | Linux |
|---|---|---|
| Browser | Microsoft Edge (comes with Windows) or Chrome 126+. Firefox also works. | Firefox, Chrome or Chromium. |
| Default output | **MP4** (H.264) | **WebM** (VP8) |
| For MP4 | Nothing extra with Edge / Chrome. With Firefox only: `ffmpeg`. | `ffmpeg`, or Chrome / Chromium 126+. |
| Display | Not needed. | Not needed. If there is no `DISPLAY`, `xvfb-run` is used automatically when it exists. |

MP4 plays almost everywhere (Windows, PowerPoint, phones, QuickTime). WebM plays in browsers and VLC.

## Quick start

### Windows (Command Prompt or PowerShell)

```bat
cd C:\path\to\git_practice\tick_viewer
py record_replay.py ..\20250213.data --biggest
py record_replay.py ..\20250213.data --at 09:30:00
```

Use `python` instead of `py` if that is how Python is started on your machine.

### Linux

```bash
cd /nfs/site/disks/km_sdmdv_1/users/jiajunka/km4/debug/practise/git_practice/tick_viewer
python3 record_replay.py ../20250213.data --biggest
python3 record_replay.py ../20250213.data --at 09:30:00
```

A run prints what it is doing and where the video went:

```
Largest print: 400 contracts @ 6135.00 at 2025-02-14 09:04:15.632 ET -> starting at 09:04:12 ET
Browser: /usr/intel/bin/firefox
Format:  webm
URL:     http://127.0.0.1:44879/?t=09:04:12&rec=10&size=1600x900&mode=rt&speed=1&fmt=webm
  viewer: recording 10s at 1600x900 (window 1366x682, video/webm;codecs=vp8) from 2025-02-14 09:04:11.870 ET
  viewer: uploading 1484393 bytes
  viewer: done
Saved .../tick_viewer/recordings/replay_20250213_090412_10s.webm (1.5 MB, WEBM recorded by the browser, 9.99 s, 22 s total)
```

A 10 s clip takes about 20-25 s in total, most of it loading the data file in the browser.

## Picking the moment

* Times are **New York time (ET)**, written `HH:MM` or `HH:MM:SS`.
* A file covers one Globex session, which starts at 18:00 ET the evening before. For
  `20250213.data`, times from `18:00` to `23:59` are on **2025-02-13** and times before `18:00`
  are on **2025-02-14**.
* `--biggest` finds the largest single print in the file and starts a little before it (30% of the
  clip, at least 1 s; change it with `--lead SECONDS`).
* A time after the end of the file starts the replay at the beginning of the file.

## Options

| Option | Default | Meaning |
|--------|---------|---------|
| `--at HH:MM[:SS]` | `09:30:00` | Start time (ET). |
| `--biggest` | | Start just before the largest print instead. |
| `--seconds N` | `10` | Length of the video. |
| `--speed X` | `1` | Replay speed. `--seconds 10 --speed 10` covers 100 s of market time. |
| `--rate` | | Treat `--speed` as prints per second instead of a multiplier. |
| `--bar N` | `200` | Candle size in ticks. |
| `--interval S` | | Time candles of 1, 5, 15 or 60 seconds instead of tick candles. |
| `--vis N` | | Number of candles on screen. |
| `--no-footprint` | | Plain candles. |
| `--no-tape` | | Leave out the Time & Sales tape. |
| `--size WxH` | `1600x900` | Video size in pixels (use even numbers). |
| `--format auto\|mp4\|webm` | `auto` | `auto` takes the extension of `-o`, else MP4 on Windows and WebM on Linux. |
| `-o FILE` | `recordings/replay_<file>_<time>_<len>s.<format>` | Output file. A `.mp4` name asks for MP4. |
| `--browser NAME\|PATH` | first one found | `edge`, `chrome`, `chromium`, `firefox`, or the path to the program. |
| `--ffmpeg PATH` | searched | `ffmpeg` to use for MP4 conversion. |
| `--timeout S` | length + 180 | Give up after this many seconds. |

Run `record_replay.py --help` for the full list.

## Examples

```bash
# 30 s around 13:33 at 5x with 5-second candles, 1920x1080
python3 record_replay.py ../20250213.data --at 13:32:30 --seconds 30 --speed 5 --interval 5 --size 1920x1080

# The opening minute, sped up 6x into 10 s, plain candles, no tape
python3 record_replay.py ../20250213.data --at 09:30:00 --speed 6 --no-footprint --no-tape

# MP4 on Linux (needs ffmpeg, see below)
python3 record_replay.py ../20250213.data --biggest -o recordings/biggest.mp4

# WebM on Windows instead of the default MP4
py record_replay.py ..\20250213.data --biggest --format webm
```

## How MP4 is produced

When MP4 is wanted, the recorder picks Edge or Chrome first, because Firefox can only record WebM.

1. **Edge / Chrome 126+** record H.264 MP4 directly. If `ffmpeg` is also available, the file is
   repackaged (no re-encoding) so every player can seek in it.
2. If the browser can't record MP4, or its MP4 encoder fails, the replay is recorded as WebM and
   converted to H.264 MP4 with `ffmpeg`.
3. If there is no `ffmpeg` either, the WebM is kept, saved under the same name with a `.webm`
   extension, and a warning says so. The run still succeeds.

`ffmpeg` is looked for in this order: `--ffmpeg PATH`, `ffmpeg` on the `PATH`, `tick_viewer/bin/ffmpeg`
(`bin\ffmpeg.exe` on Windows), then the Python package `imageio-ffmpeg` if it is installed.

### Getting ffmpeg

**Windows:** download the "essentials" build from <https://www.gyan.dev/ffmpeg/builds/>, unzip it,
and copy `bin\ffmpeg.exe` into `tick_viewer\bin\`. Alternatively, with internet access:
`py -m pip install imageio-ffmpeg`.

**Linux:** download the amd64 static build from <https://johnvansickle.com/ffmpeg/> (a single file),
copy `ffmpeg` into `tick_viewer/bin/` and `chmod +x tick_viewer/bin/ffmpeg`. Alternatively, with
internet access: `python3 -m pip install --user imageio-ffmpeg`.

## How it works

`record_replay.py` starts `view_ticks.py`'s web server on a free local port and opens
`tick_viewer.html` in a headless browser with recording options in the URL
(`?t=09:04:12&rec=10&size=1600x900&fmt=mp4...`). The page loads the `.data` file, jumps to the
start time, plays, and records itself with the browser's built-in `MediaRecorder`. It then uploads
the video back to the script, which saves it, fixes up the file (WebM length field, MP4 conversion)
and closes the browser.

Playback follows the clock, so if the machine is slow the video drops frames but still covers the
right stretch of market time.

## Watching instead of recording

```bash
python3 view_ticks.py ../20250213.data --start 09:30:00      # Linux
py view_ticks.py ..\20250213.data --start 09:30:00           # Windows
```

Then open the printed address in a browser. You can also open `tick_viewer.html` directly and drop
the `.data` file onto it.

## Troubleshooting

| Message / symptom | What to do |
|---|---|
| `no Firefox / Chrome / Edge / Chromium found` | Pass the browser with `--browser "C:\Program Files\...\msedge.exe"` (or a Linux path). |
| `timed out after ... s` | The browser could not load or record in time. Try `--timeout 600`. On Intel Linux machines the first Firefox start can be slow while it is fetched into its cache. |
| `kept the WebM instead` | MP4 was asked for but neither Edge / Chrome 126+ nor `ffmpeg` was available. See "Getting ffmpeg". |
| `Browser MP4 recording failed ...; retrying as WebM` | The browser's H.264 encoder refused (for example a very large `--size`). The retry handles it; `ffmpeg` then gives you an MP4. |
| Linux: `X11 initialization failed` | Install `xvfb-run`, or run from a session that has a `DISPLAY`. |
| A browser process is left running after you pressed Ctrl+C | Windows: end `msedge.exe` / `chrome.exe` / `firefox.exe` in Task Manager. Linux: `pkill -f tick_rec_`. |

## Tested so far

* Linux (SLES 15, Firefox 153): WebM recording, the MP4-requested fallback to WebM, and the
  `ffmpeg` hand-off.
* Windows MP4 through Edge / Chrome follows the browsers' documented MP4 support (version 126+) but
  had not been run on a Windows machine when this was written. Check the first run's
  `viewer: recording ... (video/mp4;...)` line, which confirms the browser recorded MP4 itself.
