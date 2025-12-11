import argparse
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

def export_to_onnx(model_path: str, output_path: str):
    config = VideoMAEConfig.from_pretrained(model_path)
    config._attn_implementation = "eager"
    model = VideoMAEForVideoClassification.from_pretrained(model_path, config=config)
    model.eval()

    dummy_input = torch.randn(1, 16, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    print(f"Exported ONNX model to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="output/best_model")
    parser.add_argument("--output", default="output/model.onnx")
    args = parser.parse_args()
    export_to_onnx(args.model_path, args.output)
