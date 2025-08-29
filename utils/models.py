import torch
import torch.nn as nn
import contextlib
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

# Optional: Hugging Face transformers for backbone loading
from transformers import AutoModel

def load_base_model(model_id, device='cpu'):
    """
    Loads a backbone model from Hugging Face Transformers.
    """
    return AutoModel.from_pretrained(model_id).to(device)

class ChestXrayClassifier(nn.Module):
    """
    Classification model: backbone + custom head.
    """
    def __init__(
        self,
        base_model,
        num_classes,
        freeze_base: bool = False,
        head_dims=(1024, 512),
        activation: str = "gelu",
        dropout: float = 0.2,
        bn: bool = False,
    ):
        super().__init__()
        self.base_model = base_model
        self.freeze_base = freeze_base

        act_layer = {
            "gelu": nn.GELU,
            "relu": lambda: nn.ReLU(inplace=True),
            "silu": lambda: nn.SiLU(inplace=True),
            "none": nn.Identity,
        }[activation]

        # Try to get hidden size from base_model config
        in_dim = getattr(base_model.config, "hidden_size", 768)
        layers = []
        for dim in head_dims:
            layers.append(nn.Linear(in_dim, dim))
            if bn:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(act_layer())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

        if self.freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad = False
            self.base_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base:
            self.base_model.eval()
        return self

    def forward(self, x):
        cm = torch.no_grad() if self.freeze_base else contextlib.nullcontext()
        with cm:
            out = self.base_model(x)
        # pool features (CLS token fallback)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        else:
            feats = out.last_hidden_state[:, 0, :]
        return self.classifier(feats)

class ResNetChestXray(nn.Module):
    """
    ResNet-based classifier for chest X-ray.
    Loads pretrained weights, removes last layer, adds custom head.
    """
    def __init__(self, num_classes, resnet_type='resnet18', weights='DEFAULT', head_dims=(512,), dropout=0.2, activation='relu'):
        super().__init__()
        # Select weights enum based on resnet_type
        weights_enum = None
        if resnet_type == 'resnet18':
            weights_enum = ResNet18_Weights.DEFAULT if weights == 'DEFAULT' else None
        elif resnet_type == 'resnet34':
            weights_enum = ResNet34_Weights.DEFAULT if weights == 'DEFAULT' else None
        elif resnet_type == 'resnet50':
            weights_enum = ResNet50_Weights.DEFAULT if weights == 'DEFAULT' else None

        # Load pretrained ResNet
        resnet = getattr(models, resnet_type)(weights=weights_enum)
        in_features = resnet.fc.in_features
        # Remove last layer
        self.features = nn.Sequential(*(list(resnet.children())[:-1]))  # Remove fc
        # Custom head
        act_layer = {
            "gelu": nn.GELU,
            "relu": lambda: nn.ReLU(inplace=True),
            "silu": lambda: nn.SiLU(inplace=True),
            "none": nn.Identity,
        }[activation]
        head = []
        prev_dim = in_features
        for dim in head_dims:
            head.append(nn.Linear(prev_dim, dim))
            head.append(act_layer())
            if dropout > 0:
                head.append(nn.Dropout(dropout))
            prev_dim = dim
        head.append(nn.Linear(prev_dim, num_classes))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x