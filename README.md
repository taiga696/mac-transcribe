# mac-transcribe

[English README is here](README_EN.md)

---

Macの「ダウンロード」フォルダに音声ファイルを置くだけで、自動で文字起こししてくれるローカル完結ツール。
すべて無料・オープンソース。音声データが外部サーバーに送られることはありません。

## 特徴

- 📂 **Downloadsに置くだけ** — .m4a / .wav / .mp3 / .flac / .aac を検知して自動で文字起こし開始
- 🔒 **完全ローカル処理** — 音声はMacの中だけで処理。社内会議・顧客通話の録音でも安心
- 🗣️ **話者分離対応**（オプション）— pyannote.audio で「誰が話したか」を自動判定
- ⚡ **Apple Silicon GPU で高速** — mlx-whisper 使用。5分の音声が約2〜4分（M5実測）
- 🌏 **日本語・英語自動判別**
- 💰 **完全無料** — すべてオープンソース

## 必要なもの

- macOS（Apple Silicon 推奨。Intel MacでもCPUモードで動作）
- Python 3.10以上
- 初回セットアップ時のみインターネット接続

## インストール

```bash
git clone https://github.com/taiga696/mac-transcribe.git
cd mac-transcribe
./setup.sh
```

setup.sh が以下を自動で行います：

| インストール先 | 内容 |
|---|---|
| `~/transcribe-env/` | Python仮想環境（mlx-whisper / faster-whisper / pyannote 等） |
| `~/transcribe/` | スクリプト本体 |
| `~/.local/bin/transcribe` | コマンド |
| `~/Library/LaunchAgents/` | Downloads自動監視（常駐） |

所要時間：5〜15分（ライブラリのダウンロード時間による）

## 使い方

### 自動モード（おすすめ）

音声ファイルを「ダウンロード」フォルダに置くだけ。
完了するとmacOS通知が出て、同じフォルダに `ファイル名.md` が生成されます。

### 手動モード

```bash
transcribe 音声ファイル.m4a
```

### オプション例

```bash
transcribe meeting.m4a --model medium      # 速度重視
transcribe meeting.m4a --model large-v3    # 最高精度（デフォルト）
transcribe meeting.m4a --num-speakers 2    # 話者数を指定して精度向上
transcribe meeting.m4a --no-diarize        # 話者分離なし（最速）
transcribe meeting.m4a -o notes.md         # 出力先指定
transcribe --help                          # 全オプション
```

## 話者分離の有効化（任意・無料）

「誰が話したか」を分けたい場合のみ、HuggingFaceの設定が必要です：

1. https://huggingface.co/join でアカウント作成
2. 以下2つのページで「Agree and access repository」をクリック
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. https://huggingface.co/settings/tokens で Read 権限トークンを生成
4. ターミナルで実行し、トークンを貼り付け：

```bash
~/transcribe-env/bin/hf auth login
```

未設定でも話者分離なしで文字起こしは動作します。

## 出力例（Markdown）

```markdown
**[0:00:03] SPEAKER_00**: 今日の議題は3つあります。

**[0:00:12] SPEAKER_01**: はい、確認します。
```

## トラブルシューティング

```bash
# 自動監視の状態確認
launchctl print gui/$(id -u)/local.transcribe-watcher

# ログ確認
tail -20 /tmp/transcribe-watcher.log

# 自動監視の再起動
launchctl kickstart -k gui/$(id -u)/local.transcribe-watcher
```

## 技術スタック

- [mlx-whisper](https://github.com/ml-explore/mlx-whisper) — Apple Silicon GPU 推論（デフォルト）
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CPU 推論（互換用）
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — 話者分離
- [watchdog](https://github.com/gorakhargosh/watchdog) — フォルダ監視

## ライセンス

MIT License — 自由に使って・改造して・再配布してOKです。

## 免責

本ツールは無保証です。利用によって生じたいかなる損害についても作者は責任を負いません。
