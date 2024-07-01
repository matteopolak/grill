import argparse
from typing import cast

import torch
import torch.utils.data
from PIL import Image

from dataset import transform, classes, device
from model import IngredientModel

parser = argparse.ArgumentParser(description="Test the model")
parser.add_argument("--model", type=str, required=True, help="Path to the model")
parser.add_argument("--image", type=str, required=True, help="Path to the image")

args = parser.parse_args()

model = IngredientModel(num_ingredients=len(classes)).to(device)
model.load_state_dict(torch.load(args.model))

def format_prediction(prediction: torch.Tensor) -> str:
    predicted_classes = []

    for i, v in enumerate(prediction):
        if v > 0.4:
            predicted_classes.append(classes[i])

    return ", ".join(predicted_classes)

image = Image.open(args.image)
image = cast(torch.Tensor, transform(image)).to(device)

with torch.no_grad():
    outputs = model(image.unsqueeze(0))

    for i, output in enumerate(outputs):
        print(format_prediction(output.sigmoid()))

