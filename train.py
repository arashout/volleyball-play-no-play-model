import os
import av
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score
from utils import NUM_FRAMES, read_video_pyav, sample_frame_indices

MODEL_NAME = "MCG-NJU/videomae-small-finetuned-kinetics"
IMAGE_SIZE = 224
ID2LABEL = {0: "no-play", 1: "play"}
LABEL2ID = {"no-play": 0, "play": 1}


class VideoDataset(Dataset):
    def __init__(self, root_dir, processor, split="train"):
        self.processor = processor
        self.samples = []

        root_path = Path(root_dir) / split
        for label_name in ["no-play", "play"]:
            label_dir = root_path / label_name
            if not label_dir.exists():
                print(f"Label Dir: {label_dir} does not exist, skipping")
                continue
            label_id = LABEL2ID[label_name]
            for video_path in label_dir.glob("*.mp4"):
                self.samples.append((str(video_path), label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        if total_frames == 0:
            total_frames = sum(1 for _ in container.decode(video=0))
            container.seek(0)

        indices = sample_frame_indices(NUM_FRAMES, 1, total_frames)
        video = read_video_pyav(container, indices)
        container.close()

        inputs = self.processor(list(video), return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(label)

        return inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def main():
    model_path= os.environ.get("MODEL_PATH", MODEL_NAME)
    data_dir = os.environ.get("DATA_DIR", "./data")
    output_dir = os.environ.get("OUTPUT_DIR", "./output")

    print("Using", model_path, data_dir, output_dir)
    processor = VideoMAEImageProcessor.from_pretrained(model_path)
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_dataset = VideoDataset(data_dir, processor, split="train")
    val_dataset = VideoDataset(data_dir, processor, split="val")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "best_model"))
    processor.save_pretrained(os.path.join(output_dir, "best_model"))


if __name__ == "__main__":
    main()
