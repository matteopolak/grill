import argparse

import torch
import torch.utils.data

from .dataset import RecipeImageDataset, transform, classes, device
from .model import IngredientModel

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

with torch.no_grad():
    for images, labels in dataloader:
        outputs = model(images)
        predicted = outputs

        # sigmoid
        predicted = predicted.sigmoid()

        # for each scalar in the tensor, if it's greater than 0.7 then index
        # into the `classes` list and append the class to the `predicted_classes`
        # list
        predicted_classes = []

        for i in range(len(predicted)):
            for j in range(len(predicted[i])):
                if predicted[i][j] > 0.7:
                    predicted_classes.append(classes[j])

        print("Predicted Classes:")
        print(predicted_classes)

        actual_classes = []

        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if labels[i][j] == 1:
                    actual_classes.append(classes[j])

        print("Actual Classes:")
        print(actual_classes)

        break

