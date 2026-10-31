# fal.aiで成功したKontext学習の再現

更新日: 2026-11-01。ユーザーはfal.aiでKontextを学習した際、理想に近い結果を得ている。ローカル編集・EC2学習の最初の目標は、その成功条件と画像品質を再現し、比較できるようにすること。今回新たな学習は開始していない。

## 成功回についてユーザーから得た情報

- 使用画像は **100ペア**。
- **stepsは3000** に設定。学習率の実設定はまだ分からない。
- 成功回のプロンプトは、ユーザーが **`a photo of TOK floor plan.`** と確認した。ローカルの `training_config.json` のcaption・instruction・inference_instructionを、この確認済みの文に揃えた。設定変更だけで、学習は実施していない。
- `output_lora_format` は **fal**。これはLoRAの保存時のキー・名前の形式を選ぶ設定であり、学習精度を直接高める設定ではない。ローカルで読み込むには形式の互換性を確認する。

EC2の `models/` に `M5Nrqn7TF4C94MjQIq7Ha_adapter_model.safetensors` と、fal由来に見えるconfigがある。ユーザーから「EC2に残っているLoRAが成功例だと思う」との情報を得たため、このLoRAを成功モデルの有力候補として保護する。取得済みの[config](../.local/history/20261031-191756/models/5p5Ji9mQCgce2qsbF_N0h_config_fa8cf0f5-0215-437e-bd48-387a95468e12.json)は `steps=1000`, `learning_rate=1e-4`, `default_caption=a photo of TOK floor plan.`, `output_lora_format=fal`。**ユーザーが報告した3000 stepsとは異なる**ため、この設定と上記LoRAが同じ実行の成果物とは決めつけない。モデルの形式・読み込み・画像出力と紐付けを調べる。

2026-11-01にEC2上のこの候補ファイルについて、**重み本体を読み込まずヘッダーのみ**を調べた。ファイルサイズ306,596,088 bytes、LoRA A/Bのテンソル610個、A側305個のrankはすべて **16**、保存テンソルdtypeはF32だった。計算上の格納要素数は76,628,992個。これは保存されたdtypeで、学習時の計算・optimizer dtypeは不明。キーは `base_model.model.double_blocks...` のfal形式で、融合したQKVを含み、現在のローカル実装の分割層495個とrankの意味を単純に同一視しない。

今回の情報により、成功モデルを再現する際にrank16を候補として比較する根拠ができた。ユーザーが了承したrank32も候補だが、「fal成功モデルがrank32だった」とは扱わない。Diffusers 0.37.1のコードにはfal形式の変換処理があるものの、このファイルを実際にロードして生成できることはまだ試していない。ユーザーが成功例と考えていることを尊重しつつ、実際の画像評価による確認は未完了。

## 公開APIで確認できる値

以下は2026-11-01に確認した公開APIの**既定値**であり、ユーザーが実際に成功した回で使った値とは限らない。

| 項目 | fal.aiの公開値 | 現在のEC2側候補 |
| --- | --- | --- |
| trainer | `fal-ai/flux-kontext-trainer`、FLUX.1 Kontext dev用LoRA | `FLUX.1-Kontext-dev` のLoRA |
| 学習率 | `1e-4` | `5e-5` |
| 学習steps | 既定1000。ユーザーの成功回は **3000** | 30epoch上限。batch2・累積2で概算800～850 optimizer更新。ユーザーの上限30epochを守る |
| 入力 | `*_start`・`*_end` の画像ペア。対応する `ROOT.txt` に個別編集指示が可能。なければ `default_caption` | 同様の画像ペア。現在のデータは個別captionなしと記録 |
| 推論 | LoRA endpointの既定は30 steps、guidance 2.5、resolution `match_input` | 評価は28 steps、guidance 3.5、対象図面のbucketに合わせる |

出典: [fal.ai Kontext trainerのAPI](https://fal.ai/models/fal-ai/flux-kontext-trainer/api)、[fal.ai Kontext LoRA推論API](https://fal.ai/models/fal-ai/flux-kontext-lora/api)。fal側の`steps`とこちらのoptimizer更新の厳密な内部定義や、推論実装の差は未確認。数値を一致させるだけで同じ処理になるとは限らない。

fal APIの入力スキーマでは、内部のrank、alpha、学習するencoder、対象層、optimizer、精度、LRスケジュール等が公開されていない。**falの実際のrankが32、encoder固定であったと断定しない。** ユーザーは現時点でrank32・encoder固定を候補として選択した。falの3000がoptimizer更新数と同義の場合、現行候補の約800～850更新との差は大きいが、fal内部のstep定義は未確認であり単純に同一視しない。

## 条件合わせの注意

- 成功したfalの学習回で使った画像の組、個別指示、default caption、steps、学習率、学習済みLoRA・出力configを確認する。プロンプトは `a photo of TOK floor plan.` と確認済み。ファイルが入手できれば、秘密情報を含めず重みの形式とrank、対象層を解析し、同じ入力・条件のベースと比較できる。
- fal推論時に使ったモデルendpoint、prompt、steps、guidance、LoRA強度、入力サイズとseedも確認する。公開既定値を、そのユーザーが実際に使った値と取り違えない。
- 現在のコードは、個別 `ROOT.txt` を `caption` としてCLIPに渡し、グローバル `instruction` が設定されている場合はT5にはそちらを渡す。一方falの説明では `ROOT.txt` は各画像ペアの編集指示。個別指示がある場合は条件を揃える設定・実装とテストが必要。現在のデータにはcaptionなしと記録されている。
- 現行 `VisualEvaluator.render` は画像ペアの `instruction` を使い、`inference_instruction` を独立に反映する実装ではない。学習時と推論時のpromptが異なるfal再現を行う前に対応・テストする。
- 過去の `550 step` はミニバッチ数で、optimizer更新約280回の推定。falの1000 stepsと同じだとみなさない。現在は正確なoptimizer回数をログに保存する。
- rank32とencoder固定は比較の出発点。学習可能LoRAやoptimizerのdtype、対象層、入力・正解の対応、保存後の重みの再読み込みを検証する。`TRAINING_NOTES.md` と `KONTEXT_LORA_CAPACITY_RESEARCH.md` に記録済みのBF16丸め仮説も確認対象。
- 以前のbase対旧モデルの推論はpromptが異なっていたので、その差をLoRAの効果だけとして使わない。train画像と独立したval画像を分け、同じ入力・seed・instruction・推論設定で比較する。

## 最初の再現候補

成功したfalの実際の設定や成果物をまず確認する。分からない項目だけ公開既定値を候補とし、学習率・推論条件の差を修正する。一度に多くの条件を変えず、ユーザーが1回ずつ学習投入を決める。上限30epochとGPU使用量を守りつつ、optimizer更新回数を併記し、画像で到達度を判定する。

なお、モデル選択時に使用する検証画像は完全に独立したテストにはならない。falの成功モデルとEC2学習結果の比較には、同じ未使用画像があると望ましい。
