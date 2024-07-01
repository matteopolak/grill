import argparse

import torch
import torch.utils.data
import torchvision

from dataset import RecipeImageDataset, transform, classes, device
from model import IngredientModel

parser = argparse.ArgumentParser(description='Test the model')
parser.add_argument('--model', type=str, required=True, help='Path to the model')

args = parser.parse_args()

dataset = RecipeImageDataset(
    json_file='data/annotations.json',
    img_dir='data/test/',
    transform=transform,
    partition="test"
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

model = IngredientModel(num_ingredients=len(classes)).to(device)

model.load_state_dict(torch.load(args.model))

accuracy = 0.0
total = 0

def format_prediction(prediction: torch.Tensor) -> str:
    predicted_classes = []

    for i, v in enumerate(prediction):
        if v > 0.4:
            predicted_classes.append(classes[i])

    return ', '.join(predicted_classes)

with torch.no_grad():
    for images, labels in dataloader:
        outputs = model(images)

        for i, output in enumerate(outputs):
            print("Predicted", format_prediction(output.sigmoid()))
            print("Actual", format_prediction(labels[i]))
            print("====================================")

        break

