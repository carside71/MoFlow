# データセット別の agent 軸 / 予測スタイル / Context Encoder

各データセット (NBA / ETH-UCY / SDD) でローダがどう agent 軸を扱い、それがモデルのどの context encoder と対応し、結果として「マルチエージェント結合予測」になるか「agent-centric 独立予測」になるかをまとめる。

## サマリ

| データセット | ローダ | テンソル形状 | `cfg.agents` | Context Encoder | 予測スタイル |
|---|---|---|---|---|---|
| NBA | [data/dataloader_nba.py](data/dataloader_nba.py) | `[N_scene, A=11, T, 2]` | 11 | **MTREncoder** | **マルチエージェント同時予測** (相互作用考慮) |
| ETH-UCY | [data/dataloader_eth_ucy.py](data/dataloader_eth_ucy.py) | `[N_window, 1, T, 2]` | 1 | **ETHEncoder** | **agent-centric 独立予測** |
| SDD | [data/dataloader_sdd.py](data/dataloader_sdd.py) | `[N_window, 1, T, 2]` | 1 | **ETHEncoder** | **agent-centric 独立予測** |

`N_scene` はシーン数、`N_window` は (歩行者 × 観測ウィンドウ) を平坦化した独立サンプル数。

---

## 1. データローダ — agent 軸の扱い

### NBA — マルチエージェント

[data/dataloader_nba.py:125](data/dataloader_nba.py#L125)
```python
self.actor_num = self.traj_abs.shape[1]   # = 11
```
1シーン分のテンソルが `[A=11, T, 2]` 構造を維持したままバッチ化される。モデルは `[B, 11, T, F]` を入力として受け取り、attention や context encoder がエージェント間相互作用を学習する。

### ETH-UCY — agent-centric

[data/dataloader_eth_ucy.py:108-120](data/dataloader_eth_ucy.py#L108-L120)
```python
all_data = all_data['traj']                # [A_total, T, 2]
self.all_data = torch.Tensor(all_data)
self.all_data = self.all_data[:,None,:,:]  # [A_total, 1, T, 2]   ← agent 軸を 1 に潰す
...
cfg.agents = self.all_data.shape[1]        # = 1
```
全歩行者の全観測ウィンドウを平坦リスト化し `unsqueeze(1)` で agent 軸を 1 に潰す。各サンプルは1歩行者の1ウィンドウであり、同一シーン/フレームに居合わせた他歩行者は別サンプルとしてバラバラに扱われる (同一バッチに入る保証もない)。

### SDD — agent-centric

[data/dataloader_sdd.py:143](data/dataloader_sdd.py#L143)
```python
past_traj_abs = torch.from_numpy(np.stack([scene[0] for scene in all_data], axis=0)).unsqueeze(1)  # [N, 1, T, 2]
```
ETH-UCY と同じく `.unsqueeze(1)` で agent 軸を 1 に潰す。各サンプル = 1歩行者の1ウィンドウ。

---

## 2. Context Encoder — MTREncoder vs ETHEncoder

`models/context_encoder/__init__.py` の registry で yml の `MODEL.CONTEXT_ENCODER.NAME` から選択される。

| | **MTREncoder** ([mtr_encoder.py](models/context_encoder/mtr_encoder.py)) | **ETHEncoder** ([eth_encoder.py](models/context_encoder/eth_encoder.py)) |
|---|---|---|
| **過去軌道エンコード** | `PointNetPolylineEncoder` — 時刻ごとの点を mask 付きで集約するポリラインエンコーダ ([mtr_encoder.py:32-37](models/context_encoder/mtr_encoder.py#L32-L37)) | `SocialTransformer` — `[A, P×D]` に平坦化して Linear→2層 Transformer→Linear ([eth_encoder.py:12-37](models/context_encoder/eth_encoder.py#L12-L37)) |
| **エージェント query embedding** | **チーム別専用埋め込み**: team1 (×5) / team2 (×5) / ball (×1) を連結して `[11, D]` ([mtr_encoder.py:74-85](models/context_encoder/mtr_encoder.py#L74-L85)) | 汎用 `nn.Embedding(AGENTS, D)` ([eth_encoder.py:56](models/context_encoder/eth_encoder.py#L56)) |
| **agent 軸 Transformer** | A=11 でエージェント間 self-attention が**実質的に効く** | A=1 なので外側 Transformer のエージェント間 attention は**事実上無効** |
| **入力前提** | 固定11エージェント (5+5+ball) のスポーツデータ | 任意の agent 数 (実運用では A=1) |

共通要素: `SinusoidalPosEmb` による時間 PE、`mlp_pe` で agent_query と PE を融合、最後に Transformer encoder で `[B, A, D]` を出力。

### データセットとの対応

| データセット | cfg | 使用エンコーダ |
|---|---|---|
| NBA | [cfg/nba/cor_fm.yml:41](cfg/nba/cor_fm.yml#L41), [cfg/nba/imle.yml:35](cfg/nba/imle.yml#L35) | **MTREncoder** |
| ETH-UCY | [cfg/eth_ucy/cor_fm.yml:32](cfg/eth_ucy/cor_fm.yml#L32), [cfg/eth_ucy/imle.yml:29](cfg/eth_ucy/imle.yml#L29) | **ETHEncoder** |
| SDD | [cfg/sdd/imle.yml:29](cfg/sdd/imle.yml#L29) ほか | **ETHEncoder** |

**MTREncoder は NBA 専用設計**で、team1/team2/ball という3種類の役割埋め込みを持つ点が NBA の構造 (5 vs 5 + ball) に本質的に紐づく。team query を取り除けば汎用マルチエージェントエンコーダにもなるが、現状は11人固定前提のハードコード。

**ETHEncoder は歩行者系 (ETH-UCY / SDD) 用**。これらのローダは agent 軸を 1 に潰すので、ETHEncoder のエージェント間相互作用機能は実際には働かず、`SocialTransformer` は単に `[1, P×D] → [1, D]` の MLP+self-attn として個別歩行者の時系列特徴抽出器として機能する。つまり **歩行者系では実態として「単一エージェントの過去軌道エンコーダ」**になっている。

---

## 3. トレーナ/保存パイプライン側への影響

トレーナ [trainer/denoising_model_trainers.py:416-420](trainer/denoising_model_trainers.py#L416-L420) は一貫して `[B, K, A, T*F]` のテンソルを返す。

- NBA: `A=11` のまま保持され、`.pkl` 保存時も `[B, K, 11, T, F]` 構造を維持
- ETH-UCY / SDD: `A=1` なので `.pkl` 保存時の形状は `[N, 1, T, F]` 系となり、実質「1行=1エージェントの1軌道」

評価指標も整合的に変化する:
- NBA: JADE/JFDE (joint メトリクス) が意味を持つ
- ETH-UCY / SDD: A=1 のため JADE は ADE に縮退し、agent-centric 標準の minADE₂₀ / minFDE₂₀ プロトコルと一致

---

## 4. なぜデータセットごとに違うのか

1. **ベンチマーク慣習**: ETH-UCY と SDD は Social-GAN 以降「歩行者×観測ウィンドウ単位」での minADE₂₀ / minFDE₂₀ が標準。NBA (NPSN 系) は最初から11人 joint 予測 + JADE/JFDE。SOTA と比較するため各々の慣習に合わせる必要がある。
2. **データ構造**: NBA は試合中ずっと固定の11人 → `[A=11, T, 2]` が自然。歩行者系はフレームに入退場し人数が時変なので、観測ウィンドウ完備な歩行者を1人ずつ切り出すスライディングウィンドウ方式の方が扱いやすい。
3. **相互作用の重要度**: バスケは協調・マークなど相互作用が予測本質。歩行者は弱い social-force 程度なので agent-centric でも実用上十分。
4. **評価指標の整合性**: A=1 だと JADE は ADE に縮退するため、歩行者系で joint メトリクスを測る意義が乏しい。

つまり「設計思想として切り替えた」というより、各データセット側の評価プロトコルに合わせるとローダ層およびエンコーダ選択で自然にこの違いが現れる、というのが実態。コードベース自体は両方をサポートする作りになっている。

---

## 5. マルチエージェント紐付けに関する補足 (ETH-UCY / SDD)

ETH-UCY/SDD では同一シーン/フレームのエージェント同士を結びつける ID は保存パイプライン上に残らない。サンプル保存時 ([trainer/denoising_model_trainers.py:440](trainer/denoising_model_trainers.py#L440)) の `keys_to_save` にも scene/frame index は含まれない。同一シーン内の歩行者群を復元したい場合は:

- 元データ (`data_dir/original/{subset}/{subset}_{split}.pkl`) の `traj` 配列と `past_traj_original_scale` を座標一致でマッチング
- もしくは保存ループで `indexes` キーを `keys_to_save` に追加する改修

が必要。
