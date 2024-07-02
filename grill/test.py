import argparse

import torch
import torch.utils.data
import matplotlib.pyplot as plt

from dataset import RecipeImageDataset, transform, classes, device
from model import create_model

parser = argparse.ArgumentParser(description="Test the model")
parser.add_argument("--model", type=str, required=True, help="Path to the model")
parser.add_argument("--show-images", action="store_true", help="Show images")

args = parser.parse_args()

dataset = RecipeImageDataset(
    parquet_file="data/annotations.parquet",
    img_dir="data/test/",
    transform=transform,
    partition="test"
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

model = create_model(num_ingredients=len(classes)).to(device)

model.load_state_dict(torch.load(args.model))

accuracy = 0.0
total = 0

def format_prediction(prediction: torch.Tensor) -> str:
    predicted_classes = []

    for i, v in enumerate(prediction):
        if v > 0.96:
            predicted_classes.append(f"{classes[i]} ({v:.2f})")

    return ', '.join(predicted_classes)

with torch.no_grad():
    for images, labels in dataloader:
        outputs = model(images)

        for i, output in enumerate(outputs):
            print("Predicted", format_prediction(output.sigmoid()))
            print("Actual", format_prediction(labels[i]))
            print("====================================")

            if args.show_images:
                image = images[i].permute(1, 2, 0).cpu().numpy()
                plt.xlabel(format_prediction(labels[i]))
                plt.imshow(image)
                plt.show()

