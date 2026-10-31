# EC2容量整理とGitHub退避の記録

実施日: 2026-11-01。ユーザーの依頼により、Kontext以外の学習環境を調べ、モデルを除く必要なファイルをGitHubへ退避した後、非Kontext環境と再生成可能なキャッシュを整理した。対象は従来のEC2 `i-02c9d1450ba6014f4`。

## GitHubへの保存

- [開口分類学習環境のバックアップ](https://github.com/skobayashi-glb/opening-classification-training-ec2-backup): 指定された個人アカウントの非公開リポジトリ。ソース、設定、画像、ログ、tokenizer設定などモデル重み以外480ファイルをmainに保存。commit `0a4750f6afa333ddd500c403e7c62dc40f990990`。
- [Kontextのローカル実装と操作コード](https://github.com/skobayashi-glb/kontext-training-workspace): 同じ個人アカウントの非公開リポジトリ。ローカルのコード・設定・文書70ファイルのスナップショット。モデル、データセット、認証情報、仮想環境は含めていない。
- 両リポジトリの非公開設定とpush後のcommitを確認。開口分類は、ファイルごとのSHA-256とGit上の内容を照合した。コードの元のGitHubリポジトリに追跡されていた40ファイルも保護した。
- 個人アカウントを確認する前に、EC2の既存接続先の非公開組織リポジトリにも `backup/ec2-opening-20261101-005001` ブランチを作成した。その後、個人リポジトリが当該ブランチの全ファイルを含むこと、両リモートのcommitが変わっていないことを検証し、こちらで作った重複ブランチだけを削除した。組織リポジトリのmainは変更していない。
- いずれも実行時点のスナップショットで、以後のローカル編集を自動pushする構成ではない。元の全Git履歴、仮想環境、モデル重み、認証ファイルは含めていない。

## 削除したもの

1. `/home/ec2-user/workspace/opening_classification_training`：Kontext以外の開口分類学習環境。含まれていたQwen用モデル重み2点はユーザーの指定に従いGitHubへ入れず、環境とともに削除した。ソース・画像・ログ・tokenizer設定は上記に退避済み。
2. `/home/ec2-user/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B`：開口分類用のQwenキャッシュ。
3. `/home/ec2-user/.cache/pip`：再生成可能なpipダウンロードキャッシュ。今後パッケージを入れ直す場合は再ダウンロードが必要。
4. `/home/ec2-user/.vscode-server`：稼働中のリモートVS Codeプロセスがないことを確認して削除。今後Remote SSHを再利用するときはサーバーが再インストールされる可能性がある。
5. `/home/ec2-user/workspace/flux-kontext-training/.venv`：使用していない旧Python 3.9環境。現在の `.venv312` は保持。

削除前にGPU・プロジェクトの実行中プロセス、Kontextから対象へのシンボリックリンク／Pythonパス参照、GitHub保存元とのSHA-256一致を確認した。空き容量は**24.79 GiBから46.22 GiB**へ増加し、**21.43 GiB（約23.0 GB）**を確保した。

## 保持したものと残る確認

- Kontextのデータセット、ベースモデルがあるHugging Faceキャッシュ、`runs/`、LoRA、`.venv312` を保持。削除直後に成功例候補のLoRAファイルと `.venv312/bin/python` の存在を確認した。削除後のPython import・GPU学習は未実行。
- 成功例候補 `M5Nrqn7TF4C94MjQIq7Ha_adapter_model.safetensors` は、ヘッダー上でrank16・保存F32・305層。実際の画像評価と、3000stepsの成功回との対応は未確認。別にある1000stepsのconfigと同一回とは断定しない。
- `/opt/pytorch` 等はシステム側の共有環境で、依存が未確認のため保持した。Kontext用キャッシュは削除しない。
- インスタンスは `stopped` を確認済み。今回、モデルの新規学習は開始していない。

取得した容量内訳と照合記録は `.local/storage-inspection/` に保存。ユーザーのファイルの保護のため、モデルの削除は開口分類環境内とQwenキャッシュに限定した。
