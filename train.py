import argparse
import logging

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data

from dataset import RecipeImageDataset, transform, classes, device, class_weights
from model import IngredientModel

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(prog="train.py", description="Fire up the grill")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train the model")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a model checkpoint to load (in the checkpoints/ directory)")
parser.add_argument("--plot", action="store_true", help="Plot the loss")

args = parser.parse_args()

num_epochs = args.epochs
batch_size = args.batch_size

dataset = RecipeImageDataset(
    json_file='data/annotations.json',
    img_dir='data/train/',
    transform=transform,
    partition="train"
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = IngredientModel(num_ingredients=len(classes)).to(device)

if args.checkpoint is not None:
    model.load_state_dict(torch.load(f"checkpoints/{args.checkpoint}"))

criterion = nn.BCEWithLogitsLoss(weight=class_weights).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

plot_loss = []

# add plot title and labels
plt.title("Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")

for epoch in range(num_epochs):
    epoch_loss = 0.0
    collected_loss = 0.0

    for i, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        epoch_loss += loss.item()
        collected_loss += loss.item()

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            logger.info(f'Epoch {epoch}, Batch {i}, Loss: {collected_loss/10}')

            if args.plot:
                plot_loss.append(loss.item())
                plot_loss = plot_loss[-100:]

                plt.clf()
                plt.plot(plot_loss)
                plt.pause(0.05)

            collected_loss = 0.0

    logger.info(f'Epoch {epoch}, Loss: {epoch_loss/len(dataloader)}')
    logger.info(f"Saving model to checkpoints/grill-epoch{epoch}.pth")

    torch.save(model.state_dict(), f"checkpoints/grill-epoch{epoch}.pth")

    accuracy = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            predicted = torch.round(outputs)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    logger.info(f'Epoch {epoch}, Accuracy: {accuracy/total}')

torch.save(model.state_dict(), "models/grill.pth")

