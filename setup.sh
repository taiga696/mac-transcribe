#!/bin/zsh
# setup.sh — ローカル文字起こし環境 一括インストーラ
#
# 使い方:
#   1. このフォルダ（transcribe-portable）を新しいMacにコピー
#   2. ターミナルで  cd transcribe-portable && ./setup.sh
#
# インストール先:
#   ~/transcribe-env/        Python venv（ライブラリ群）
#   ~/transcribe/            スクリプト本体
#   ~/.local/bin/transcribe  コマンド（symlink）
#   ~/Library/LaunchAgents/local.transcribe-watcher.plist  自動監視

set -e
cd "${0:A:h}"

echo "=== 1/6 環境チェック ==="
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: このスクリプトは macOS 専用です" >&2
    exit 1
fi
ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then
    echo "  Apple Silicon ($ARCH): mlx-whisper (GPU) が使えます"
else
    echo "  Intel Mac ($ARCH): mlx-whisper は使えません。faster-whisper (CPU) で動作します"
fi

# Python 3.10+ を探す
PYFOUND=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then
        v=$("$c" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor:02d}")' 2>/dev/null || echo 0)
        if [[ "$v" -ge 310 ]]; then
            PYFOUND="$(command -v $c)"
            break
        fi
    fi
done
if [[ -z "$PYFOUND" ]]; then
    echo "ERROR: Python 3.10 以上が見つかりません" >&2
    echo "       以下のどちらかでインストールしてください:" >&2
    echo "         - https://www.python.org/downloads/ (公式インストーラ)" >&2
    echo "         - brew install python@3.13" >&2
    exit 1
fi
echo "  Python: $PYFOUND ($($PYFOUND --version))"

echo "=== 2/6 venv 作成 (~/transcribe-env) ==="
if [[ ! -x "$HOME/transcribe-env/bin/python" ]]; then
    "$PYFOUND" -m venv "$HOME/transcribe-env"
fi
VENV_PIP="$HOME/transcribe-env/bin/pip"
"$VENV_PIP" install -q --upgrade pip

echo "=== 3/6 ライブラリ インストール（数分かかります）==="
PKGS=(faster-whisper pyannote.audio torch torchaudio av watchdog huggingface_hub)
if [[ "$ARCH" == "arm64" ]]; then
    PKGS+=(mlx-whisper)
fi
"$VENV_PIP" install -q "${PKGS[@]}"
echo "  完了"

echo "=== 4/6 スクリプト配置 (~/transcribe, ~/.local/bin) ==="
mkdir -p "$HOME/transcribe" "$HOME/.local/bin"
cp transcribe.py watch_and_transcribe.py "$HOME/transcribe/"
cp transcribe "$HOME/transcribe/transcribe"
chmod +x "$HOME/transcribe/transcribe" "$HOME/transcribe/transcribe.py"
ln -sf "$HOME/transcribe/transcribe" "$HOME/.local/bin/transcribe"
echo "  完了"

# PATH 登録（未登録なら ~/.zshrc に追記）
if ! grep -q '.local/bin' "$HOME/.zshrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
    echo "  PATH を ~/.zshrc に追加しました"
fi

echo "=== 5/6 Downloads 自動監視 (LaunchAgent) ==="
PLIST_DST="$HOME/Library/LaunchAgents/local.transcribe-watcher.plist"
mkdir -p "$HOME/Library/LaunchAgents"
sed -e "s|__HOME__|$HOME|g" transcribe-watcher.plist.template > "$PLIST_DST"
launchctl bootout "gui/$(id -u)/local.transcribe-watcher" 2>/dev/null || true
if launchctl bootstrap "gui/$(id -u)" "$PLIST_DST" 2>/dev/null; then
    echo "  自動監視を有効化しました（Downloads に音声を置くと自動で文字起こし）"
else
    echo "  ※ 自動監視の即時有効化に失敗しました。Mac を再ログインすれば自動で有効になります"
fi

echo "=== 6/6 動作テスト ==="
say -o /tmp/transcribe_setup_test.m4a "セットアップのテストです。これが表示されれば成功です。" 2>/dev/null || true
if [[ -f /tmp/transcribe_setup_test.m4a ]]; then
    "$HOME/.local/bin/transcribe" /tmp/transcribe_setup_test.m4a --no-diarize --model tiny -o /tmp/transcribe_setup_test.md \
        && echo "  ✓ テスト成功: /tmp/transcribe_setup_test.md" \
        || echo "  ※ テスト失敗。transcribe /tmp/transcribe_setup_test.m4a --no-diarize --model tiny を手動で試してください"
fi

cat <<'EOF'

========================================
 セットアップ完了！
========================================

【使い方】
  transcribe 音声.m4a            # 手動で文字起こし
  Downloads に音声ファイルを置く  # 自動で文字起こし（.md が生成される）

【話者分離を有効にするには（任意・無料）】
  1. https://huggingface.co/join でアカウント作成
  2. 以下2つのページで「Agree and access repository」:
     - https://huggingface.co/pyannote/speaker-diarization-3.1
     - https://huggingface.co/pyannote/segmentation-3.0
  3. https://huggingface.co/settings/tokens で Read 権限トークンを生成
  4. ターミナルで:
       ~/transcribe-env/bin/hf auth login
     （トークンを貼り付け）

  設定後は自動的に話者分離（SPEAKER_00 等のラベル）付きで出力されます。
  未設定でも文字起こし自体は動きます。

【初回実行時の注意】
  モデル（約3GB）を初回のみダウンロードします。Wi-Fi 環境でどうぞ。
EOF
