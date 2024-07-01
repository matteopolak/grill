import argparse
import logging
import os

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data

from dataset import RecipeImageDataset, transform, classes, device, pos_class_weights
from model import create_model

logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOG", "INFO").upper()
)

parser = argparse.ArgumentParser(prog="train.py", description="Fire up the grill")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train the model")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate)")
parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint to resume training")
parser.add_argument("--plot", action="store_true", help="Plot the loss")

args = parser.parse_args()

epoch_start = args.checkpoint + 1 if args.checkpoint is not None else 0
num_epochs = args.epochs
batch_size = args.batch_size

dataset = RecipeImageDataset(
    parquet_file="data/annotations.parquet",
    img_dir="data/train/",
    transform=transform,
    partition="train"
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = create_model(num_ingredients=len(classes)).to(device)

if args.checkpoint is not None:
    model.load_state_dict(torch.load(f"checkpoints/grill-epoch{args.checkpoint}.pth"))

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_class_weights).to(device)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)

plot_loss = []

# add plot title and labels
plt.title("Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")

logger.info(f"{(len(dataset) + batch_size - 1) // batch_size} batches per epoch")

for epoch in range(epoch_start, num_epochs + epoch_start):
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
            logger.info(f"Epoch {epoch}, Batch {i}, Loss: {collected_loss/10}")

            if args.plot:
                plot_loss.append(loss.item())
                plot_loss = plot_loss[-100:]

                plt.clf()
                plt.plot(plot_loss)
                plt.pause(0.05)

            collected_loss = 0.0

    logger.info(f"Epoch {epoch}, Loss: {epoch_loss/len(dataloader)}")
    logger.info(f"Saving model to checkpoints/grill-epoch{epoch}.pth")

    torch.save(model.state_dict(), f"checkpoints/grill-epoch{epoch}.pth")

    accuracy = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            predicted = outputs.sigmoid()

            total += labels.size(0)
            accuracy += (predicted - labels).abs().sum().item()

    logger.info(f"Epoch {epoch}, Accuracy: {accuracy/total}")

torch.save(model.state_dict(), "models/grill.pth")

