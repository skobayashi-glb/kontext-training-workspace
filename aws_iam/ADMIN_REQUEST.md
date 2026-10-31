# Kontext学習用のIAM設定依頼案

## 状況

AWSアカウント末尾1887の東京リージョンで、学習のたびにGPU EC2を作成し、データ・モデル・結果を専用S3バケットに保存する予定です。既存の他用途のEC2やIAMロールは使用しません。

専用バケット `kontext-training-340752821887-3fc34c1f` は作成済みで、パブリックアクセスはブロック済みです。まだデータは転送していません。

現在のIdentity Centerの `PowerUserAccess` セッションでは、EC2作成のドライランは成功しましたが、`iam:PassRole` が拒否されました。管理者権限で新しい作業用ロールと、そのロールをEC2に渡す限定的な権限を用意してください。**AdministratorAccess全体を利用者に付与する必要はありません。**

## 依頼する設定

1. EC2が引き受けるIAMロール `KontextTrainingWorker` を作成し、同名のインスタンスプロファイルを作成してください。信頼ポリシー案は [worker-trust.json](worker-trust.json) です。
2. ロールに [worker-policy.json](worker-policy.json) を設定してください。専用バケットの一覧・読み取り、`jobs/` への結果書き込み、および東京リージョンで `KontextManaged=ephemeral-training` タグを持つEC2の終了に限ります。S3オブジェクトの削除権限は含みません。
3. 学習を投入するユーザーのIdentity Center権限セットまたは専用のアクセス権に、[launcher-pass-role.json](launcher-pass-role.json) の `iam:PassRole` を付与してください。対象は上記ロールだけで、EC2サービスへの受け渡しに限定します。組織の権限セット変更が他の利用者に影響する場合は、専用の権限セットや割当方法を検討してください。
4. S3への経路、暗号化キー、SCP・権限境界など、組織側で追加の制限がある場合は、それに合わせて設定してください。EC2の作成・タグ付与・終了とS3への読み書きが可能か、後でドライラン・小さな検証を行います。

EC2側の終了権限は上記タグの**全インスタンス**に適用されます。「自身のインスタンスだけ」にIAMだけで限定される設計ではありません。専用ロールの利用対象・タグ付け権限・アプリ側のID検証について管理者にも確認をお願いします。成果物保存を検証してから、そのジョブで作成したIDだけを終了する制御は、別途実装・テストします。

この資料は設定案で、IAMロール・権限セットはまだ作成・変更していません。3つのJSONはAWS IAM Access Analyzerの構文・ポリシー検証で指摘なしですが、実際の権限や組織のポリシーとの整合は未検証です。設定後も、このPCのランチャーと実環境での一連の動作確認が必要です。

参考: [EC2へのIAMロール受け渡し](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/permission-to-pass-iam-roles.html)、[タグ条件付きEC2操作](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ExamplePolicies_EC2.html)。
