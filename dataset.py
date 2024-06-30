import os
import pandas as pd

import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms

# resize to 128x128, normalize to [0, 1], and convert to tensor
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

classes = pd.read_json("data/annotations.json").explode("ingredients")["ingredients"].unique().tolist()
# print non-strings
print(list(filter(lambda x: not isinstance(x, str), classes)))
classes = sorted(classes)

print(classes)

class RecipeImageDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        self.annotations = pd.read_json(json_file)

        self.img_dir = img_dir
        self.transform = transform

        self.paths = os.listdir(img_dir)
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        id = path.split('.')[0]

        img_path = os.path.join(self.img_dir, path)
        image = read_image(img_path)

        annotation = self.annotations.query(f"id == '{id}'").to_dict(orient="records")[0]

        image_classes = annotation['ingredients'].tolist()
        # 1 for each class in the recipe's ingriedients list, 0 otherwise
        label = torch.tensor([1.0 if c in image_classes else 0.0 for c in classes])

        if self.transform:
            image = self.transform(image)

        return image, label

