import torch.nn as nn
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

def create_model(num_ingredients: int) -> nn.Module:
    model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_ingredients)

    return model

