# dance2mmd

ダンス動画（YouTube Shorts くらいの尺）から振り付けをボーンとして抽出し、
**中間ボーンデータ(JSON)** を介して MMD の `.vmd` モーションを生成する。
ボーン抽出を「継ぎ目」にして、抽出器（AI）を差し替えられるのが設計の肝。

```
動画 ──[1]姿勢推定──▶ ★中間ボーンJSON ──┬─[3a]── dance.vmd  (MMD/Blender/three.js)
   backend A: MediaPipe (Mac/CPU)        └─[3b]── three.js プレビュー (即確認)
   backend B: WHAM/GVHMR (GPUサーバ)
```

中間JSONは人が読める共通フォーマット（`src/dance2mmd/skeleton.py` の canonical skeleton）。
どの抽出器も「このJSONを吐く」だけが仕事。`.vmd` 変換（SMPL/関節→MMDボーン）は
AIに投げず自前で持つ（`src/dance2mmd/retarget.py`）。

## 同梱しないもの（ライセンス）
著作物はリポジトリに含めません。各自で用意してください：
- **3Dモデル(.pmx)＋テクスチャ** → `models/`（例: HoYoverse配布のMMDモデル。再配布不可のため非同梱）
- **入力ダンス動画 / 抽出した .vmd・中間JSON・音声** → `data/`（楽曲・動画の著作権のため非同梱。`data/sample_motion.json` の合成データのみ同梱）

## セットアップ
```bash
pip install -e .                 # コア（numpy のみ）
pip install -e ".[mediapipe]"    # Mac ローカル抽出を使うなら（mediapipe + opencv）
pip install -e ".[dev]" && pytest -q   # テスト
```

## 使い方（Mac だけで完結する確認ループ）
```bash
# 1. 動画 → 中間ボーンJSON（MediaPipe, GPU不要）
dance2mmd extract dance.mp4 -o data/dance.json --smooth 5 --ground

# 2. ブラウザで振り付けを即確認（抽出が合ってるかを目視）
dance2mmd view data/dance.json
#   ↑ うまく file:// で読めない場合:
#   python -m http.server  → http://localhost:8000/viewer/index.html に
#   data/dance.json をドラッグ＆ドロップ

# 3. MMD モーション生成
dance2mmd retarget data/dance.json -o data/dance.vmd --scale 12.5 --model "モデル名"
```

動画なしで全体を試す（合成モーション）:
```bash
python -c "from dance2mmd.synthetic import make; make(120).to_json('data/sample.json')"
dance2mmd view data/sample.json
```

## 高精度版（GPUサーバ）
`docs/GPU_SETUP.md` 参照。WHAM/GVHMR で `joints.npy`(SMPL 24関節) を作り、
`extract_wham.from_smpl_joints()` で **同じ中間JSON** にする。以降 [2][3] は共通。

## 現状の精度と既知の制約（v1）
- リターゲットは **swing-only FK**（ねじり/twist は復元しない）。腕のロールは出ない。
- ターゲットモデルは **Tポーズ前提**。Aスタンスのモデルは腕にオフセットが要る。
- 脚は FK 駆動（足IK・接地拘束は未実装。滑り・めり込みは起こりうる）。
- 単眼推定なので奥行き(Z)は弱い。横移動の大きい振り付けほど誤差が乗る。

### 初回キャリブレーションのコツ
1. まず **viewer** で canonical JSON が「立っていて・正面を向いている」か確認
   （違えば WHAM 側 `up=` を調整。MediaPipe は補正済み）。
2. `.vmd` をモデルに乗せて再生。鏡像/反転していたら
   `src/dance2mmd/retarget.py` の `_to_mmd_quat`（Z軸ハンドednessの反転）を疑う。
   ここ1行が座標系変換の単一ポイント。
3. 身長スケールは `--scale` で合わせる（1m ≈ 12.5 MMD単位が目安）。

## ロードマップ
- [ ] 足IK + 接地（足滑り/めり込み解消）
- [ ] twist 復元（前腕ロール等）
- [ ] three.js 側で MMDLoader による実モデル＋vmd 再生（[3b]の発展）
- [ ] GVHMR バックエンド追加
- [ ] 複数人トラッキング → 人物選択

## レイアウト
```
src/dance2mmd/
  skeleton.py        canonical skeleton（全体の契約）
  motion.py          中間フォーマット Motion + JSON I/O
  extract_mediapipe.py  backend A（Mac）
  extract_wham.py       backend B（GPU: SMPL関節→canonical）
  postprocess.py     平滑化・接地
  quat.py            クォータニオン
  retarget.py        ★ canonical → MMDボーン → VMD（自前ロジック）
  vmd.py             VMD バイナリ書き出し
  synthetic.py       動画なしテスト用モーション
  cli.py             dance2mmd コマンド
viewer/index.html    three.js プレビュー（中間JSONを再生）
docs/GPU_SETUP.md    GPUサーバでの WHAM 手順
```
