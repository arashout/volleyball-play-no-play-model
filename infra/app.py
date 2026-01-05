#!/usr/bin/env python3
import aws_cdk as cdk
from stack import VolleyballPredictStack

app = cdk.App()
VolleyballPredictStack(app, "VolleyballPredictStack")
app.synth()
