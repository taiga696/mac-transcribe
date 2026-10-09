#!/usr/bin/env python3
"""
watch_and_transcribe.py — Downloads の音声ファイルを監視して自動文字起こし

新しい .m4a / .wav / .mp3 / .flac / .aac が Downloads に置かれたら、
録音完了を待ってから自動で transcribe を実行する。
出力は WorkBuddy の outputs ディレクトリに保存。

Usage:
    python watch_and_transcribe.py             # フォアグラウンドで実行
    # バックグラウンド常駐化は local.transcribe-watcher.plist 参照

Settings (環境変数で上書き可能):
    WATCH_DIR  監視ディレクトリ (default: ~/Downloads)
    OUT_DIR    出力先 (default: ~/Downloads)
    TRANSCRIBE transcribe コマンドのパス (default: ~/.local/bin/transcribe)
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# --- Config ---
WATCH_DIR = Path(os.environ.get("WATCH_DIR", Path.home() / "Downloads")).expanduser()
# 出力先を Downloads にすることで、元音声と同じ場所に .md が生成される
OUT_DIR = Path(os.environ.get("OUT_DIR", Path.home() / "Downloads")).expanduser()
TRANSCRIBE = os.environ.get("TRANSCRIBE", str(Path.home() / ".local/bin/transcribe"))
AUDIO_PATTERN = re.compile(r'\.(m4a|wav|mp3|flac|aac)$', re.IGNORECASE)


def _ts() -> str:
    return time.strftime('%H:%M:%S')


def _notify(title: str, message: str, sound: str = "default") -> None:
    """macOS 通知を表示（osascript 経由）。失敗は無視。"""
    title = title.replace("'", "'\\''")
    message = message.replace("'", "'\\''")
    script = (
        f"display notification '{message}' "
        f"with title '{title}' sound name '{sound}'"
    )
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
    except Exception:
        pass


class AudioHandler(FileSystemEventHandler):
    """Handle new audio files in the watched directory."""

    def _wait_until_stable(self, p: Path) -> bool:
        """Wait until the file size stops changing (export/copy finished).

        QuickTime exports and large copies make the file visible long before
        it is complete; opening it mid-export can even fail with EPERM.
        Returns True if the file exists and is stable, False if it vanished.
        """
        last_size = -1
        stable_count = 0
        deadline = time.time() + 10 * 60  # give up waiting after 10 min
        while time.time() < deadline:
            if not p.exists():
                print(f"[{_ts()}] File vanished: {p}", flush=True)
                return False
            size = p.stat().st_size
            if size == last_size and size > 0:
                stable_count += 1
                if stable_count >= 3:  # unchanged for 3 consecutive checks
                    return True
            else:
                stable_count = 0
            last_size = size
            time.sleep(5)
        print(f"[{_ts()}] WARNING: file size still changing after 10 min, "
              f"proceeding anyway", flush=True)
        return True

    def _run_transcribe(self, p: Path) -> None:
        """Run transcribe on p with retries (transient EPERM etc.)."""
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = OUT_DIR / f"{p.stem}.md"
        log = Path("/tmp") / f"transcribe-{p.stem}.log"
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            print(f"[{_ts()}] Transcribing → {dest} (attempt {attempt})",
                  flush=True)
            _notify("文字起こし開始", f"{p.name} を処理中…", sound="Blow")
            try:
                # 話者分離ありで実行（HF トークン未設定時は transcribe 本体が自動フォールバック）
                result = subprocess.run(
                    [TRANSCRIBE, str(p), "-o", str(dest)],
                    capture_output=True, text=True, timeout=3 * 60 * 60,  # 3h 上限
                )
                log.write_text(
                    f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}\n",
                    encoding="utf-8",
                )
                if result.returncode == 0:
                    print(f"[{_ts()}] ✓ Done: {dest}", flush=True)
                    _notify("文字起こし完了", f"{p.name} → {dest.name}", sound="Glass")
                    return
                print(f"[{_ts()}] ✗ Failed (rc={result.returncode}, "
                      f"attempt {attempt}/{max_attempts}). See {log}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"[{_ts()}] ✗ Timed out (>3h). Aborted.", flush=True)
                _notify("タイムアウト", f"{p.name} が3時間超過で中断", sound="Basso")
                return
            except Exception as e:
                print(f"[{_ts()}] ✗ Error (attempt {attempt}): {e}", flush=True)
            if attempt < max_attempts:
                time.sleep(60)  # backoff before retry
        _notify("文字起こし失敗", f"{p.name} の処理に失敗（{max_attempts}回試行）",
                sound="Basso")

    def on_created(self, event):
        # ディレクトリは無視
        if event.is_directory:
            return
        src = event.src_path
        if not AUDIO_PATTERN.search(src):
            return
        print(f"[{_ts()}] Detected: {src}", flush=True)
        # 録音・エクスポート完了を待つ（ファイルサイズが安定するまで）
        print(f"[{_ts()}] Waiting for file to stabilize…", flush=True)
        p = Path(src)
        if not self._wait_until_stable(p):
            return
        self._run_transcribe(p)

    def on_moved(self, event):
        # Finder からのドラッグ&ドロップは move イベントで来る
        # dest_path が新しいパス
        if event.is_directory:
            return
        dest_path = getattr(event, "dest_path", None)
        if not dest_path:
            return
        # 一時的に on_created と同じ処理を実行する
        if not AUDIO_PATTERN.search(dest_path):
            return
        print(f"[{_ts()}] Detected (moved): {dest_path}", flush=True)
        # on_created と同じフローを使うためダミーイベント発火
        class _E:
            pass
        e = _E()
        e.is_directory = False
        e.src_path = dest_path
        self.on_created(e)


def main():
    # 二重起動防止（LaunchAgent と手動/screen 起動が競合しないように）
    import fcntl
    lock_path = Path("/tmp/transcribe-watcher.lock")
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[{_ts()}] Another watcher instance is already running. Exit.",
              flush=True)
        return

    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{_ts()}] Watching: {WATCH_DIR}", flush=True)
    print(f"[{_ts()}] Output to: {OUT_DIR}", flush=True)
    print(f"[{_ts()}] Transcribe: {TRANSCRIBE}", flush=True)
    print(flush=True)

    handler = AudioHandler()
    obs = Observer()
    obs.schedule(handler, str(WATCH_DIR), recursive=False)
    obs.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[{_ts()}] Stopping…", flush=True)
        obs.stop()
    obs.join()


if __name__ == "__main__":
    main()
