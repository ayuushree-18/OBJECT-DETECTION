from typing import List, Tuple
from PIL import Image
import torch
import torchvision.transforms as T

# ImageNet normalization
_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def crop_and_resize_patches(pil_image: Image.Image,
                            boxes: List[Tuple[int,int,int,int]],
                            size=(224,224)) -> Tuple[torch.FloatTensor, List[Tuple[int,int,int,int]]]:
    """
    Returns:
      patches_tensor: torch.FloatTensor shape (N,3,H,W)
      mapping_boxes: list of boxes same order as patches
    """
    patches = []
    img_w, img_h = pil_image.size
    for (x1,y1,x2,y2) in boxes:
        # clip to image bounds
        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(img_w, x2); y2c = min(img_h, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        crop = pil_image.crop((x1c, y1c, x2c, y2c)).convert("RGB")
        patch = _transform(crop)
        patches.append(patch)
    if len(patches) == 0:
        return torch.empty((0,3,size[0],size[1])), []
    patches_tensor = torch.stack(patches, dim=0)
    return patches_tensor, boxes[:len(patches)]
