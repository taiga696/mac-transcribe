# mac-transcribe

[日本語版 README はこちら](README.md)

---

Just drop an audio file into your Mac's **Downloads** folder and it gets transcribed automatically. A fully local tool — no data ever leaves your machine. Free and open source.

## Features

- 📂 **Just drop it in Downloads** — detects `.m4a` / `.wav` / `.mp3` / `.flac` / `.aac` and starts transcribing automatically
- 🔒 **100% local processing** — audio never leaves your Mac. Safe for internal meetings and customer calls
- 🗣️ **Speaker diarization** (optional) — automatically identifies "who spoke when" with pyannote.audio
- ⚡ **Fast on Apple Silicon GPU** — powered by mlx-whisper. A 5-minute recording transcribes in ~2–4 minutes (measured on M5)
- 🌏 **Auto-detects Japanese & English**
- 💰 **Completely free** — all open source

## Requirements

- macOS (Apple Silicon recommended; Intel Macs run in CPU mode)
- Python 3.10+
- Internet connection (only for the first-time setup)

## Installation

```bash
git clone https://github.com/taiga696/mac-transcribe.git
cd mac-transcribe
./setup.sh
```

`setup.sh` handles everything automatically:

| Location | Contents |
|---|---|
| `~/transcribe-env/` | Python virtual environment (mlx-whisper / faster-whisper / pyannote, etc.) |
| `~/transcribe/` | Main scripts |
| `~/.local/bin/transcribe` | Command |
| `~/Library/LaunchAgents/` | Downloads auto-watcher (daemon) |

Expected time: 5–15 minutes (depending on library download speed)

## Usage

### Auto mode (recommended)

Just drop an audio file into your **Downloads** folder. When done, a macOS notification appears and a `filename.md` file is generated in the same folder.

### Manual mode

```bash
transcribe audio.m4a
```

### Options

```bash
transcribe meeting.m4a --model medium      # faster, lower accuracy
transcribe meeting.m4a --model large-v3    # highest accuracy (default)
transcribe meeting.m4a --num-speakers 2    # improve accuracy by specifying speaker count
transcribe meeting.m4a --no-diarize        # skip diarization (fastest)
transcribe meeting.m4a -o notes.md         # specify output path
transcribe --help                          # all options
```

## Enabling speaker diarization (optional, free)

Only needed if you want to know "who spoke when":

1. Create an account at https://huggingface.co/join
2. On each of the following pages, click "Agree and access repository":
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Generate a **Read** token at https://huggingface.co/settings/tokens
4. Run the following in the terminal and paste your token:

```bash
~/transcribe-env/bin/hf auth login
```

Diarization works without these settings — it just runs without speaker labels.

## Sample output (Markdown)

```markdown
**[0:00:03] SPEAKER_00**: We have three items on the agenda today.

**[0:00:12] SPEAKER_01**: Got it, let me confirm.
```

## Troubleshooting

```bash
# Check the auto-watcher status
launchctl print gui/$(id -u)/local.transcribe-watcher

# Check the log
tail -20 /tmp/transcribe-watcher.log

# Restart the auto-watcher
launchctl kickstart -k gui/$(id -u)/local.transcribe-watcher
```

## Tech stack

- [mlx-whisper](https://github.com/ml-explore/mlx-whisper) — Apple Silicon GPU inference (default)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CPU inference (fallback)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization
- [watchdog](https://github.com/gorakhargosh/watchdog) — folder monitoring

## License

MIT License — free to use, modify, and redistribute.

## Disclaimer

This tool is provided "as is" without warranty. The author is not liable for any damages arising from its use.
