import torch
from PIL import Image
import numpy as np

from src.model import build_model, predict_batch, LABELS
from src.warp_and_crop import crop_and_resize_patches
from src.proposals import get_selective_search_proposals, filter_proposals

if __name__ == "__main__":
    model, device = build_model(device="cpu")

    img = Image.open("data/sample.jpg").convert("RGB")

    # Generate a few proposals
    boxes = get_selective_search_proposals(img)
    boxes = filter_proposals(boxes, max_proposals=5)

    # Crop + resize proposals to 224×224
    patches, mapped = crop_and_resize_patches(img, boxes)

    print("Patch batch shape:", patches.shape)

    # Predict softmax probabilities
    probs = predict_batch(model, patches, device, batch_size=5)

    print("\nPredictions:")
    for i, p in enumerate(probs):
        print(f"Box {i} ->", {LABELS[j]: float(p[j]) for j in range(len(LABELS))})
