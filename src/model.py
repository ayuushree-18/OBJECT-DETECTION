import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# CLASS ORDER (must remain consistent everywhere)
LABELS = ['background', 'person', 'bicycle', 'car']

def build_model(num_classes: int = 4, device: str = "cpu"):
    """
    Builds a ResNet18 backbone with a small classification head.
    Returns:
      model: torch.nn.Module
      device: torch.device
    """
    dev = torch.device(device)

    # Load pretrained ResNet18 backbone
    backbone = models.resnet18(pretrained=True)

    # Get number of features from the last FC layer
    in_features = backbone.fc.in_features

    # Replace the final FC layer with our own classifier head
    backbone.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    backbone.to(dev)
    backbone.eval()
    return backbone, dev


def predict_batch(model, patches_tensor, device, batch_size: int = 32):
    """
    patches_tensor: torch.FloatTensor (N,3,224,224)
    Returns:
      numpy array shape (N, num_classes) of softmax probabilities.
    """
    import numpy as np

    # Handle empty case
    if patches_tensor is None or patches_tensor.shape[0] == 0:
        return np.zeros((0, len(LABELS)))

    model.to(device)
    model.eval()
    probs_list = []

    with torch.no_grad():
        N = patches_tensor.shape[0]
        for i in range(0, N, batch_size):
            batch = patches_tensor[i : i + batch_size].to(device)
            logits = model(batch)              # (batch_size, num_classes)
            probs = F.softmax(logits, dim=1)   # softmax
            probs_list.append(probs.cpu().numpy())

    return np.vstack(probs_list)