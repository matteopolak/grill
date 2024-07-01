import torch
import torch.nn as nn
import torch.utils.data
from dataset import RecipeImageDataset, transform, classes, device, class_weights
from model import IngredientModel
from matplotlib import pyplot as plt

dataset = RecipeImageDataset(
    json_file='data/annotations.json',
    img_dir='data/train/',
    transform=transform,
    partition="train"
)

N = len(classes)

num_epochs = 5
batch_size = 64

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = IngredientModel(num_ingredients=N).to(device)

#model.load_state_dict(torch.load("checkpoints/model-epoch0.pth"))

criterion = nn.BCEWithLogitsLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("batches per epoch:", (len(dataset) + batch_size - 1) // batch_size)

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

        plot_loss.append(loss.item())

        if i % 10 == 0:
            print(f'Epoch {epoch}, Batch {i}, Loss: {collected_loss/10}')
            collected_loss = 0.0

            # clear old stuff
            plt.clf()
            plt.plot(plot_loss)
            plt.pause(0.05)

    print(f'Epoch {epoch}, Loss: {epoch_loss/len(dataloader)}')

    print(f"saving model to checkpoints/model-epoch{epoch}.pth")
    torch.save(model.state_dict(), f"checkpoints/model-epoch{epoch}.pth")
    print("model saved")

    accuracy = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            predicted = torch.round(outputs)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}, Accuracy: {accuracy/total}')

torch.save(model.state_dict(), "trained/model.pth")

