import os
import pickle
import polars as pl

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# resize to 256x256, normalize to [0, 1], and convert to tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

with open("data/classes.pkl", "rb") as f:
    # dict[str, int] -- class name to recipes/count
    c = pickle.load(f)

    classes = sorted(c.keys())
    pos_class_weights = torch.tensor([c[cls] for cls in classes])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RecipeImageDataset(Dataset):
    def __init__(self, parquet_file, img_dir, transform=None, partition="train"):
        self.annotations = (pl.read_parquet(parquet_file)
            .filter(pl.col("partition") == partition)
            .with_row_index())

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations.row(index, named=True)
        id = annotation["id"]

        img_path = os.path.join(self.img_dir, id[:1], id[1:2], id[2:3], id[3:4], id + ".jpg")
        image = Image.open(img_path)

        image_classes = annotation["ingredients"]
        # 1 for each class in the recipe's ingriedients list, 0 otherwise
        label = torch.tensor([1.0 if c in image_classes else 0.0 for c in classes])

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image.to(device), label.to(device)

