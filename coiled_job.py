import sys
import boto3
import tempfile
from pathlib import Path
from export_predictions import export_predictions

MODEL_PATH = "/app/model.onnx"

def main():
    bucket, key = sys.argv[1], sys.argv[2]
    s3 = boto3.client("s3")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "video.mp4"
        output_path = Path(tmpdir) / "predictions.json"

        s3.download_file(bucket, key, str(video_path))
        export_predictions(MODEL_PATH, str(video_path), str(output_path))

        result_key = key.replace("uploads/", "results/").replace(".mp4", "_predictions.json")
        s3.upload_file(str(output_path), bucket, result_key)
        print(f"Uploaded results to s3://{bucket}/{result_key}")

if __name__ == "__main__":
    main()
