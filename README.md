# ローカル編集・EC2 GPU学習

コードはこのPCで編集し、学習時だけ既存EC2を使います。環境やデータを再インストールせず、実行ごとの専用フォルダーへ最新コードを送り、学習完了・エラー・キャンセル後にEC2を停止します。EC2を起動するだけでは学習しません。

## 最初の準備

このフォルダーを **Windows側のVS Code** で開きます。編集する学習コードは `flux-kontext-training/src/flux_kontext_training/` です。EC2から取得した元のコードと開発用UIも `flux-kontext-training/` にあります。

```powershell
python -m pip install -r requirements-local.txt
aws login --profile default
```

WindowsのAWS CLI v2とOpenSSH、既存のSSH設定 `samurai-AWS` を使います。AWS認証はCLIに任せるため、ランチャーへアクセスキーを登録する必要はありません。SSHの秘密鍵やAWS認証ファイルは転送しません。

`aws_config.json` に既存インスタンスと確認済みパスを設定済みです。`host_key_alias` は、既にknown_hostsで信頼されている元のEC2アドレスをホスト鍵の識別に使っています。接続先IPはAWSから毎回取得します。ホスト鍵の検証でエラーになる場合は、正しい鍵を確認してください。検証を無効にする設定は使いません。

## 通常の操作

1. `flux-kontext-training/src/flux_kontext_training/` のコードを編集して保存します。
2. `training_config.json` でデータセット、instruction、epoch数などを指定します。初期値は既存環境を使うための例です。学習に使うcaption・instructionを含め、実行前に内容を確認してください。
3. 同期予定を確認し、学習を投入します。

```powershell
# AWSにもSSHにもアクセスしません。EC2を起動しません。
python aws_run.py train --dry-run

# 起動 → コード同期 → 学習投入。完了時にEC2側で停止します。
python aws_run.py train

# 別の学習設定を使う場合
python aws_run.py train --config my_training_config.json
```

「学習ジョブを受け付けました」と表示されたら、ターミナルやVS Codeを閉じても、PCをオフにしても学習はEC2で継続します。受付前は起動・転送中なので、その表示を待ってください。

学習時間を15分に制限する処理はありません。6分のタイマーは投入失敗時の起動放置を防ぐためのもので、systemdが学習ジョブを実行中なら停止しません。学習中のハングには自動の強制終了時間を設けていないため、必要なら下記のcancelを使います。

既存のPython環境は `/home/ec2-user/workspace/flux-kontext-training/.venv312/bin/python` です。既存データとmodelsを使い、元のコード・学習履歴は上書きしません。`.git`、仮想環境、画像、データ、重み、過去のrunsは同期対象から除外します。

## 結果・ログ・lossを見る

```powershell
# EC2を起動せず、取得済み結果をローカルで閲覧します。
python aws_run.py view
```

ブラウザーで **http://127.0.0.1:8765** を開きます。ジョブの状態、train/validation loss、実行ログ、学習設定を表示します。表示の更新やこの画面の起動ではEC2を起動しません。

PCがオンラインなら、学習投入後にローカルのバックグラウンド取得プロセスが約10秒ごとにログとmetricsを保存します。通常は最終結果を取得した後にEC2が停止します。PCがオフでも、EC2は終了結果を保存し、最大90秒の取得待ち後に停止します。

PCがオフだった間の最終ログは、EC2のEBSに残ります。**次の学習でEC2を起動したときに、未取得の前回ログを自動で補完します。** ログ取得だけのために追加起動する必要はありません。次の学習を待たずに最終結果が必要な場合は、下記の `fetch --start` で短時間だけ起動して取得し、取得後に停止します。

```powershell
# 末尾のジョブIDは省略すると直近ジョブを使います。
python aws_run.py fetch --start

# 学習済み重みもこのPCへダウンロード
python aws_run.py fetch --start --weights

# EC2が既に起動している場合、起動せずに取得
python aws_run.py fetch
```

ローカル保存先は `.local/results/<ジョブID>/` です。重みは明示的に `--weights` を指定したときに取得します。PCがオフだった結果も、次回の学習でEC2を起動した際に `fetch` で取得できます。

```powershell
python aws_run.py status                 # AWSの状態確認。起動しません。
python aws_run.py jobs                   # ローカルのジョブ一覧。AWS接続なし。
python aws_run.py logs                   # 取得済みログ。AWS接続なし。
python aws_run.py logs --follow          # ローカルログの更新を追跡
python aws_run.py cancel                 # 学習をキャンセルして終了・自動停止
```

古いジョブの操作は、たとえば `python aws_run.py fetch <ジョブID> --start` のように指定できます。

## 任意の学習スクリプトを実行する

スクリプトをローカルの `flux-kontext-training/` に置いて保存します。引数はシェルを通さずに渡します。

```powershell
python aws_run.py run --dry-run -- python my_train.py --epochs 10
python aws_run.py -- python my_train.py --epochs 10
```

コマンドが終了した時点でEC2を停止するため、常駐APIサーバーではなく、学習終了時にプロセスが終了するコマンドを指定します。既存アプリ用の `train` コマンドは、この条件を満たすCLIを使っています。

## EC2側の実行と停止

- ジョブは `~/.kontext-remote/jobs/<ジョブID>/` に作られます。コードはその中の `workspace/`、ログは `log.txt`、状態は `status.json` です。
- 学習の設定・metrics・重みは `workspace/runs/<学習ID>/` に保存されます。`training.json` に学習IDと出力先を記録します。以前からある元プロジェクトの `runs/` は変更しません。
- systemdの `kontext-train-<ジョブID>.service` がSSHとは独立して学習を実行します。起動時に有効化される常駐学習サービスは作りません。
- 学習コマンドの成功・失敗・キャンセルを記録してから終了します。systemdの `ExecStopPost` で停止するため、ワーカーが異常終了しても停止処理を実行します。
- インスタンスのOSシャットダウン動作が `stop`、ルートがEBSであることを投入前に確認します。terminateへの変更は行いません。
- 同じ起動中の重複投入を拒否し、GPU上に別プロセスがある場合も学習を開始しません。
- 既存のチェックポイントから再開するときは、`source_checkpoint_id` に元の学習IDを指定します。元のweightsを参照し、新しい実行の結果は別ディレクトリへ保存します。

学習途中のキャンセルは処理の区切りで反映されます。既存学習コードは各epochで良い検証結果の重みを保存する方式です。未完了epochの状態まで保存する保証はありません。

## 障害時

認証切れの場合は `aws login --profile default` を実行します。学習が受付済みなら、ローカルのAWS認証が切れてもEC2側の学習と停止処理は継続します。

受付時にSSHが切れて実行状況が不明になった場合は、実行中の学習を誤停止しないため、ランチャーから無条件に停止しません。`status` で確認してください。起動直後の準備失敗には独立したローカル監視とEC2側タイマーを用意しています。

停止の要求から `stopped` までは数分かかる場合があります。次の `train` は前回の停止完了を待ってから起動します。手動停止が必要なら次を使います。実行中の学習も止まるので、対象ジョブを確認して実行してください。

```powershell
aws ec2 stop-instances --instance-ids i-02c9d1450ba6014f4 --region ap-northeast-1 --profile default
```

EC2を停止しても、環境・データ・結果を保持するEBSの保存料金は残ります。通常停止の `stopping` 状態ではインスタンス使用料は課金されません。起動ごとに最低1分のインスタンス使用料があります。[AWSの状態別課金説明](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html)

## 学習の画像評価と、終了後の自動診断

`training_config.json` は、画像評価付きLoRA学習の提案設定へ更新しています。上限30epoch、固定乱数の検証、2epochごとの複数画像比較、停滞時の学習率低下・早期停止を含みます。次の学習を自動で再投入する機能はありません。毎回の診断を読んで、次の投入はご自身で決めます。

```powershell
# 保存済みのモデルを画像評価するだけ。学習・重み更新はしません。
python aws_run.py evaluate

# 1回の学習を投入。完了後は画像・lossを自動診断します。
python aws_run.py train

# 取得済み結果の診断を手動で再実行。EC2は起動しません。
python aws_run.py review --force
```

画像は `view` の「画像評価」で、入力・正解・ベース・学習後を比較できます。wall IoU等は補助指標であり、形状や開口を画像でも確認します。検証画像はモデル選定にも使うため、この結果だけを独立テスト精度とは呼びません。

学習用画像2枚は、学習前と選択したモデルでの終了時の変換を保存します。valの定期評価とは分け、学習画像も再現できないのかを確認できます。少数のtrain画像だけで過学習の有無は断定しません。

最終結果の取得後、ローカルのCodex CLIを読み取り専用で呼び出し、`.local/results/<ジョブID>/review/ai_review.md` に日本語の診断と次回の改善案を保存します。レビューではシェル実行・ブラウザー操作を無効にし、EC2起動・学習投入・設定変更を行わない指示を渡します。レビューはEC2とは独立しており、Codexの利用枠を使います。現在のPCではCLIとChatGPTログインを確認済みです。公式仕様: [Codexの非対話モード](https://learn.chatgpt.com/docs/non-interactive-mode)。

PCがオフなら、結果がローカルへ取得できた後に診断します。ログイン切れ等で診断が失敗した場合は `review/state.json` に理由を残し、勝手に何度も再試行しません。再実行は `python aws_run.py review --force` を使います。このチャットへ勝手に新しいメッセージを送る構成ではなく、結果画面とレポートへ診断を保存する構成です。

調査の更新先は [TRAINING_NOTES.md](TRAINING_NOTES.md) です。認証、過去の実設定、TOK、LoRA容量、凍結対象、LR、損失、過学習の確認状況をまとめています。診断が成功すると末尾の自動欄を最新レポートへのリンクに更新します。自動診断の制御はCPUのモックテスト済みで、実モデルを使った無人レビュー全体は未検証です。

## 検証コマンド

```powershell
python -m unittest discover -s tests -v
python aws_run.py train --dry-run
```

学習成功・失敗・キャンセル、設定エラー時の起動防止、投入状況不明時の停止抑制、ログ保存後の停止通知、同期対象、過去ジョブの保護、結果閲覧をGPUなしで検証します。
