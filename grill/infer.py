import argparse
from typing import cast

import torch
import torch.utils.data
from PIL import Image

from dataset import transform, classes, device
from model import create_model

parser = argparse.ArgumentParser(description="Test the model")
parser.add_argument("image", type=str, help="Path to the image")
parser.add_argument("--model", type=str, default="models/grill.pth", help="Path to the model")
parser.add_argument("--confidence", type=float, default=0.9, help="Confidence threshold")

args = parser.parse_args()

model = create_model(num_ingredients=len(classes)).to(device)
model.load_state_dict(torch.load(args.model))

def format_prediction(prediction: torch.Tensor) -> str:
    predicted_classes = []

    for i, v in enumerate(prediction):
        if v >= args.confidence:
            predicted_classes.append(f"{classes[i]} ({v:.2f})")

    return ", ".join(predicted_classes)

image = Image.open(args.image)
image = cast(torch.Tensor, transform(image)).to(device)

with torch.no_grad():
    model.eval()
    outputs = model(image.unsqueeze(0))

    for i, output in enumerate(outputs):
        print(output)
        print(format_prediction(output.sigmoid()))

