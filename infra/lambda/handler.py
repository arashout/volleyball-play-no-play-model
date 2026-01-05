import json
import os
import urllib.request
import urllib.error
import boto3

def get_coiled_token():
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=os.environ["COILED_SECRET_NAME"])
    return response["SecretString"]

def handler(event, context):
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    key = record["s3"]["object"]["key"]

    token = get_coiled_token()
    software_env = os.environ["COILED_SOFTWARE_ENV"]
    model_bucket = os.environ["MODEL_BUCKET"]
    model_key = os.environ["MODEL_KEY"]

    script = f"""
import boto3
import tempfile
import json
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
import threading
import queue

NUM_FRAMES = 16
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LABELS = ["no_play", "play"]

def preprocess_frame(frame):
    h, w = frame.shape[:2]
    scale = min(224 / w, 224 / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((224, 224, 3), dtype=np.uint8)
    top = (224 - new_h) // 2
    left = (224 - new_w) // 2
    padded[top:top + new_h, left:left + new_w] = resized
    arr = padded.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)

def run_inference(session, frames):
    processed = np.stack([preprocess_frame(f) for f in frames])
    pixel_values = processed[np.newaxis, ...].astype(np.float32)
    outputs = session.run(None, {{"pixel_values": pixel_values}})
    logits = outputs[0]
    pred_idx = int(np.argmax(logits, axis=-1)[0])
    confidence = float(np.exp(logits[0, pred_idx]) / np.exp(logits[0]).sum())
    return LABELS[pred_idx], confidence

def frame_reader(cap, frame_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_queue.put(frame_rgb)

bucket = "{bucket}"
key = "{key}"
model_bucket = "{model_bucket}"
model_key = "{model_key}"

s3 = boto3.client("s3")
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    video_path = tmpdir / "video.mp4"
    model_path = tmpdir / "model.onnx"
    output_path = tmpdir / "predictions.json"

    print("Downloading model...")
    s3.download_file(model_bucket, model_key, str(model_path))
    print("Downloading video...")
    s3.download_file(bucket, key, str(video_path))

    session = ort.InferenceSession(str(model_path))
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window_frames = NUM_FRAMES
    stride_frames = NUM_FRAMES // 2
    window_duration = window_frames / fps

    predictions = []
    frame_buffer = []
    frame_idx = 0
    window_start_frame = 0

    frame_queue = queue.Queue(maxsize=32)
    stop_event = threading.Event()
    reader_thread = threading.Thread(target=frame_reader, args=(cap, frame_queue, stop_event), daemon=True)
    reader_thread.start()

    pbar = tqdm(total=total_frames, desc="Analyzing")
    while True:
        frame_rgb = frame_queue.get()
        if frame_rgb is None:
            break
        frame_buffer.append(frame_rgb)
        frame_idx += 1
        if len(frame_buffer) >= NUM_FRAMES:
            indices = np.linspace(0, len(frame_buffer) - 1, NUM_FRAMES, dtype=int)
            sampled = [frame_buffer[i] for i in indices]
            label, confidence = run_inference(session, sampled)
            start_time = window_start_frame / fps
            end_time = start_time + window_duration
            predictions.append({{
                "startTime": round(start_time, 3),
                "endTime": round(end_time, 3),
                "label": "no-play" if label == "no_play" else "play",
                "confidence": round(confidence, 4)
            }})
            frame_buffer = frame_buffer[stride_frames:]
            window_start_frame += stride_frames
        pbar.update(1)

    stop_event.set()
    pbar.close()
    cap.release()

    output = {{"playNoPlayPredictions": predictions}}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    result_key = key.replace("uploads/", "results/").replace(".mp4", "_predictions.json")
    s3.upload_file(str(output_path), bucket, result_key)
    print(f"Uploaded {{len(predictions)}} predictions to s3://{{bucket}}/{{result_key}}")
"""

    payload = json.dumps({
        "script": script,
        "vm_type": "g4dn.xlarge",
        "software": software_env,
        "region": "us-east-1",
    }).encode()

    req = urllib.request.Request(
        "https://cloud.coiled.io/api/v2/jobs/single-job-script/",
        data=payload,
        headers={
            "Authorization": f"ApiToken {token}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        raise Exception(f"Coiled API error: {e.code} - {error_body}")

    sns = boto3.client("sns")
    sns.publish(
        TopicArn=os.environ["SNS_TOPIC_ARN"],
        Subject="Volleyball Prediction Job Started",
        Message=f"Job submitted for s3://{bucket}/{key}\nCoiled response: {json.dumps(result, indent=2)}",
    )

    return {"statusCode": 200, "body": json.dumps(result)}
