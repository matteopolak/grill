import torch
import torch.nn as nn
import torch.utils.data
from dataset import RecipeImageDataset, transform, classes
from model import IngredientModel

dataset = RecipeImageDataset(
    json_file='data/annotations.json',
    img_dir='data/images/train',
    transform=transform,
    partition='train'
)

N = len(classes)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

model = IngredientModel(num_ingredients=N)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

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

        if i % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {collected_loss/10}')
            collected_loss = 0.0

    accuracy = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            predicted = torch.round(outputs)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}, Accuracy: {accuracy/total}')

