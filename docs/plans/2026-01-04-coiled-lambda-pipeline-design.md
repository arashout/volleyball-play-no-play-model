# Coiled + Lambda Video Prediction Pipeline

## Overview

S3-triggered Lambda that invokes Coiled GPU jobs for video prediction.

## Architecture

```
[S3 uploads/] → [Lambda] → [Coiled REST API] → [GPU Job] → [S3 results/]
                    ↓
                 [SNS Topic]
```

## Components

| Component | Purpose |
|-----------|---------|
| S3 Bucket | `uploads/` for input videos, `results/` for predictions |
| Lambda | Triggered by S3, calls Coiled REST API |
| Secrets Manager | Stores Coiled API token |
| SNS Topic | Job submission notifications |
| Coiled Environment | `volleyball-predict` with model baked in |

## Files to Create

```
infra/
├── app.py              # CDK entry point
├── stack.py            # Infrastructure definition
├── lambda/
│   └── handler.py      # Lambda function
└── requirements.txt    # CDK deps

coiled_job.py           # Script executed on Coiled GPU
```

## Coiled Environment

Name: `volleyball-predict`

Dependencies:
- opencv-python-headless
- onnxruntime-gpu
- numpy
- tqdm
- boto3

Includes: `best_model/model.onnx` → `/app/model.onnx`

## AWS Profile

Use `--profile personal` for all AWS/CDK commands.

## Deployment Order

1. Create Coiled environment with model
2. Store Coiled token in Secrets Manager
3. `cdk deploy --profile personal`
4. Test with sample upload
