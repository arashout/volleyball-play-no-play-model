from aws_cdk import (
    Stack,
    RemovalPolicy,
    Duration,
    CfnOutput,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_sns as sns,
    aws_secretsmanager as secretsmanager,
    aws_s3_notifications as s3n,
    aws_iam as iam,
)
from constructs import Construct

class VolleyballPredictStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        bucket = s3.Bucket(
            self, "VideoBucket",
            removal_policy=RemovalPolicy.RETAIN,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        topic = sns.Topic(self, "JobNotificationTopic")

        secret = secretsmanager.Secret.from_secret_name_v2(
            self, "CoiledToken", "coiled-api-token"
        )

        handler = lambda_.Function(
            self, "TriggerCoiledJob",
            runtime=lambda_.Runtime.PYTHON_3_11,
            code=lambda_.Code.from_asset("lambda"),
            handler="handler.handler",
            timeout=Duration.seconds(30),
            environment={
                "SNS_TOPIC_ARN": topic.topic_arn,
                "COILED_SECRET_NAME": "coiled-api-token",
                "COILED_SOFTWARE_ENV": "volleyball-predict",
                "MODEL_BUCKET": bucket.bucket_name,
                "MODEL_KEY": "models/model.onnx",
            },
        )

        secret.grant_read(handler)
        topic.grant_publish(handler)
        bucket.grant_read(handler)

        handler.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:PutObject"],
            resources=[bucket.bucket_arn, f"{bucket.bucket_arn}/*"],
        ))

        bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(handler),
            s3.NotificationKeyFilter(prefix="uploads/", suffix=".mp4"),
        )

        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "SnsTopicArn", value=topic.topic_arn)
