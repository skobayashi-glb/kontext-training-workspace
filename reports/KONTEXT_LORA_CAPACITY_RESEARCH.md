# Kontextの学習パラメータ・凍結範囲の調査

更新: 2026-11-01。ユーザーの主な疑問は、学習画像を再現できない理由が、学習可能パラメータの数や位置、凍結範囲にあるのではないかというもの。今回は公開資料・ローカルのコードとモデル構造・CPU上の小さな数値実験を確認した。EC2の起動・学習は行っていない。

## 調査結果の要点

- 現在案のrank 32は、この実装では**約1.72億個**のTransformer LoRAパラメータ。過去のrank 64は約**3.44億個**。同じrankでもLoRA対象の層を広げるとパラメータ数が増え、他の実装のrankだけとの比較はできない。
- 既存のrank 64で良い変換が得られなかったことは、容量不足の可能性を完全には否定しない。学習率・実更新数・精度・入力と正解の整合・推論でのadapter適用などが同時に影響しており、**rankのみを増やした比較実験はまだない**。
- 過去の `550 step` 等はこのコードでは主に**ミニバッチの回数**。勾配累積が2のため、optimizerの更新はそれより少ない。実例に照らすと、おおむね280回／248回であり、以前の報告ではこの区別が十分でなかった。
- モデル内部の学習可能パラメータがBF16の場合、小さい更新が丸められる可能性がある。小さなCPU実験では再現したが、**過去のEC2学習時の実際のadapter dtypeや、何％の重みが更新されなかったかは未測定**。保存ファイルのdtypeだけでは学習中のdtypeを断定しない。

## 公開されているKontextの設定例

これらは実装や研究者が示した例で、採用頻度の統計や、建築図面での最適値ではない。

| 出典 | rank・学習率の例 | 学習対象と条件 |
| --- | --- | --- |
| [Hugging Face DiffusersのKontext I2I例](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md) | rank 16、LR 1e-4、最大1000 optimizer step、batch 1・勾配累積4 | ペア画像と編集instructionを使用。現在の[公式スクリプト](https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux_kontext.py)のI2I分岐は `train_text_encoder` 無効を要求。例は最新版であり、EC2のDiffusers 0.37.1とそのまま互換とは限らない |
| [Musubi TunerのKontext例](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/flux_kontext.md) | network dim 32、LR 1e-4、16 epoch | Transformer LoRA、エンコーダー出力の事前計算、BF16等の例。コードや対象層は本環境と同一とは限らない |
| [100組の対応画像を使ったKontext研究](https://openaccess.thecvf.com/content/WACV2026W/FoMoV/papers/Clouser_Few-Shot_LoRA_Adaptation_of_a_Flow-Matching_Foundation_Model_for_Cross-Spectral_WACVW_2026_paper.pdf) | rank 16・32、alpha=rank、LR 1e-4・5e-4、1000 step以上を比較 | TransformerのattentionとMLPにLoRA、元の重みは固定。別の画像変換タスクであり、図面への直接的な最適値ではない |

現在のDiffusers公式KontextスクリプトはattentionとFF等を対象にする例。本環境はさらにnorm系・一部出力層等も対象とする。対象層を変えればrankが同じでもパラメータ数は変わる。[Diffusersの対象層の説明](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/README_flux.md)

## 今の設定で何個学習されるか

Diffusers 0.37.1の `FluxTransformer2DModel` を**重みを読み込まないmetaデバイス**上に構築し、本プロジェクトの対象層選択コードを適用した。使用したのは19個のdual block、38個のsingle block、幅3072、guidance embeddingsありの標準構造。基準Transformer全体は11,901,408,320個で、biasを含む。EC2上の元モデルconfigを直接再読込したわけではなく、標準構造が一致することを前提とした計算。

本実装の選択範囲は495の線形層（dual 266、single 228、最終出力1）。学習されるのは各層に追加するLoRAのA・B行列であり、元の各層の重みは固定される。計算式は各層について `rank × (入力次元 + 出力次元)` の和。現行設定ではalpha=rank。

| rank | Transformer LoRAの学習可能数 | 元のTransformer全体に対する比率 | BF16のLoRA重みだけの容量 |
| --- | ---: | ---: | ---: |
| 16 | 85,967,872 | 約0.72% | 約164 MiB |
| 22（過去A） | 118,205,824 | 約0.99% | 約225 MiB |
| 32（現在案） | 171,935,744 | 約1.44% | 約328 MiB |
| 64（過去B） | 343,871,488 | 約2.89% | 約656 MiB |
| 128 | 687,742,976 | 約5.78% | 約1,312 MiB |

過去Bの保存ファイルは687,878,720バイト、LoRA_A層数は495。rank64・BF16の計算上の重み本体687,742,976バイトと比べ、差は135,744バイトで、ファイルヘッダー等のオーバーヘッドと整合する。これは構造の一致を強く支持するが、厳密な元モデルconfigと全テンソルdtypeの実測とは区別する。

公式Kontextスクリプトのデフォルト対象名を同じ構造に当てはめた計算では380線形層。rank16は約5,417万個、rank32は約1.08億個。**本実装のrank32約1.72億個とは対象層が異なる**。公式スクリプトの最新版とこちらの実装の差もあるため、この計算は固定した対象名に基づく比較であり、品質比較ではない。

rankだけでなくalphaによる強さの違いにも注意する。現在の本実装はalpha=rankで倍率1。一方、調査時点のDiffusers I2IのREADME例はrank16を指定しalphaを明示せず、同じ時点の[スクリプトのalpha既定値](https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux_kontext.py)は4なので、読み合わせ上はalpha/rank=0.25。スクリプトやREADMEは将来変わりうる。

optimizerの既定値にも差がある。現在の本実装は `torch.optim.AdamW` をweight_decay指定なしで呼ぶため、PyTorchの既定値0.01を使用する。上記のDiffusers公式スクリプトの既定値は1e-4。この差だけが不調の原因とは分からないが、比較実験では対象層・rank・alpha・LRに加えて記録すべき。現時点のtraining_config.jsonにはweight_decayを明示する項目がない。

`reports/lora_capacity_analysis.json` は計算結果、`.local/analyze_lora_capacity.py` は再計算スクリプト。

## どこを固定し、どこを学習するか

| 部分 | 今の画像評価付き設定 | 過去 |
| --- | --- | --- |
| Transformer元重み | 固定 | A/Bとも固定 |
| Transformer LoRA | 学習、rank32 | Aは実rank22、Bは実rank64 |
| CLIP | 固定 | AはCLIP LoRAを学習、Bは固定 |
| T5 | 固定 | AはT5 LoRAを学習、Bは固定 |
| VAE | 固定 | A/Bとも固定 |

入力画像由来の情報と文章条件を扱うTransformerのattention・FF・一部norm等へLoRAを入れている。TE側を固定しても文章は条件として使用される。旧Aと旧Bはencoderだけでなくrank・LR・損失も異なるため、encoderの効果は単独で判断できない。

**実装上の制約:** 現在の画像評価・早期停止付き `VisualExperiment` は、CLIP/T5の学習を有効にすると拒否する。基礎trainerにはencoder LoRAのコードがあるが、画像評価・保存・再読み込みまで含む比較経路は準備できていない。以前の「全ての学習設定を変更できる」という説明は、この制約について不正確だった。今回、対応外の設定はEC2を起動する前のローカル検証で拒否するようにした。encoderも比較対象にするなら、保存・再読み込み・同条件画像評価を先に実装・検証する。

Transformerの全重みを学習する案は、約119億パラメータ。仮に重み・勾配・Adamの2状態をすべてBF16で持つという単純計算でも約88.7 GiBで、ベースモデル以外やactivationを含めない。このEC2のL40S約45 GiBでは通常の全層学習は難しい。CPUオフロードなどの手段はあるが別途設計と速度検証が必要。

参考としてLoRAの重み・勾配・Adamの2状態をすべてFP32で持つ場合の単純計算はrank32で約2.56 GiB、rank64で約5.12 GiB、rank128で約10.25 GiB。これはLoRA分だけで、固定モデル・VAE・encoder・activation・一時領域を含まない。これだけからrank128がGPUに収まるとは判断できない。

## 学習率とBF16の検証

コードではGPUがBF16に対応すると、ベースモデルをBF16でロードし、その後LoRAを付ける。実EC2で使用したPEFTバージョンとadapterの実dtypeは、この時点で確認できていない。

ローカルの小型の実Tensorを使い、同じadapter取り付け関数を呼んだところ、Diffusers 0.37.1 / PEFT 0.17.1 / PyTorch 2.7.0のCPU環境では、BF16 baseに追加した42個のtrainable tensorもBF16だった。これは**このローカル構成での観察**で、EC2でも必ず同じとは限らない。meta device上のadapter dtypeは実体ありのdeviceと異なるため、dtypeの判断には使わない。

別の単一パラメータのCPU実験では、初期値約0.01・一定勾配・AdamW・LR 1e-5のBF16値が10更新後も同じだった。FP32では同条件で約0.0100→0.0099に変化した。重み0.01付近のBF16間隔が、提案された1回の更新より大きいことで起こる**一例**。LR 5e-5と1e-4の例ではBF16値も変化した。実際の各重み・勾配・optimizer stateは分布しているので、これだけで本学習の更新停止や原因を断定できない。結果は `reports/bf16_toy_update.json`、実行は `.local/test_bfloat16_update.py`。

保存されたLoRA Bが非ゼロだったことは、少なくとも一部が変わった強い証拠だが、別のA行列の全要素や各stepの実効更新量までは証明しない。次にEC2を使う際はGPUの実環境で**A/Bごとのdtype、勾配ノルム、optimizer状態のdtype、代表サンプルのstep前後の実際の変化量・変化しない割合**を記録したい。BF16計算を維持しながら学習可能LoRAとoptimizer状態をFP32にする案は候補だが、VRAM使用量とPEFT・保存形式の互換性を実測してから採用する。Diffusers公式スクリプトは少なくともFP16設定では学習可能LoRAをFP32に上げる処理を持つが、BF16について同じ処理を常に行うとは述べない。[公式ソース](https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux_kontext.py)

## 過去のstep数と候補実験

過去Aは10epochで記録 `550` ミニバッチ、batch 2・勾配累積2、各epoch55ミニバッチ。各epoch最後の端数でも更新する実装なら、およそ `ceil(55/2)×10 = 280` optimizer更新。過去Bは8epoch×55ミニバッチ＋途中48ミニバッチで計488、同じ前提では約 `28×8+24 = 248` 更新。ログの記録だけから当時の中断直前のoptimizer状態まで実測したわけではないため、あくまで推定。

公開例の1000 optimizer stepと、こちらの記録550 mini-batchをそのまま比較できない。画像の延べ提示数、累積回数、LR、ノイズ、スケジュールも揃える必要がある。今の30epoch上限では、109組・batch2・累積2なら概算800～850更新程度だが、bucketの端数や早期停止で変動する。

実験案は、費用を抑えるためユーザーが1回ずつ選ぶ。いずれもtrain画像とval画像を別に固定し、同じprompt・seed・推論条件で比較する。

1. **追加学習なしの確認:** 既存LoRAをtrain入力で推論し、同じpromptのベース・正解・適用後を比較する。チェックポイントのロード成功と重みが有効なことも確認。前回はvalだけを見ており、trainを確認したとは言えない。
2. **更新可能性と少数例の診断:** 条件を固定し、選んだ少数組で拡張なし。A/Bのdtype・実更新、train出力を確認する。少数データは1epochあたりの更新回数が減るため、30epochだけで「学べない」と結論しない。
3. **容量の比較:** 比較する場合は、固定対象・prompt・dtype・学習率・optimizer更新回数・データ・乱数等を揃え、たとえばrank32と64、必要なら64と128を別々の学習で比較する。過去Bの64と新設定32は条件が違うため、直接的なrank比較に使えない。rank128はVRAMと計算時間の事前確認が必要。
4. **encoderの比較:** 段階3と分けて、Transformer LoRAを固定条件にしたうえでCLIP/T5 LoRAの有無を比較する。ただし画像評価・保存・読み込みの対応とメモリ検証を先に行う。

次の実行や変更を最適と断定せず、ユーザーが1回ごとに判断する。過学習・一般化については、train数枚だけでなく未使用の図面での評価も必要。
