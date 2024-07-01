import torch
import torch.nn as nn
import torch.utils.data
from dataset import RecipeImageDataset, transform, classes, device
from model import IngredientModel

dataset = RecipeImageDataset(
    json_file='data/annotations.json',
    img_dir='data/val/',
    transform=transform,
    partition="val"
)

N = len(classes)

num_epochs = 5
batch_size = 64

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = IngredientModel(num_ingredients=N).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("batches per epoch:", (len(dataset) + batch_size - 1) // batch_size)

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
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {collected_loss}')
            collected_loss = 0.0

    torch.save(model.state_dict(), f"checkpoints/model-epoch{epoch}.pth")


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

