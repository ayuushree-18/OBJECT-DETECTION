import io
from typing import List, Dict

from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from src.utils import draw_boxes_on_image
from src.utils import nms  # you already have an NMS util

# Map COCO class IDs to the labels you want to DISPLAY
# 1=person, 2=bicycle, 3=car in COCO
ID_TO_NAME = {
    1: "human",     # show "human" (your XML label)
    2: "bicycle",
    3: "car",
}

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None

_transform = T.Compose([
    T.ToTensor(),   # Faster R-CNN expects tensors in [0,1]
])

def _load_model():
    global _model
    if _model is None:
        # Pretrained Faster R-CNN on COCO
        _model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        _model.to(_device)
        _model.eval()
        print("Loaded pretrained Faster R-CNN on", _device)
    return _model

def _pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def predict_faster_from_bytes(
    image_bytes: bytes,
    score_thresh: float = 0.6
):
    """
    Run pretrained Faster R-CNN and return:
      detections: list of {label, score, box}
      image_with_boxes_bytes: PNG bytes

    Post-processing:
      - keep only classes person/bicycle/car
      - for each class, do NMS
      - keep ONLY the single best box per class (like your XML)
    """
    model = _load_model()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(img).to(_device)

    with torch.no_grad():
        out = model([tensor])[0]

    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    labels = out["labels"].cpu().numpy()

    # group boxes per class
    per_class: Dict[str, List] = {}
    for box, score, lab in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        if int(lab) not in ID_TO_NAME:
            continue
        name = ID_TO_NAME[int(lab)]
        per_class.setdefault(name, []).append((box, float(score)))

    detections = []
    for cls_name, items in per_class.items():
        if not items:
            continue
        cls_boxes = [it[0] for it in items]
        cls_scores = [it[1] for it in items]
        # NMS to remove duplicates
        keep_idx = nms(
            [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in cls_boxes],
            cls_scores,
            iou_thresh=0.5,
        )
        # take only the SINGLE best box among the kept ones
        if not keep_idx:
            continue
        # choose index with highest score among keep_idx
        best = max(keep_idx, key=lambda i: cls_scores[i])
        x1, y1, x2, y2 = cls_boxes[best]
        detections.append({
            "label": cls_name,
            "score": cls_scores[best],
            "box": [int(x1), int(y1), int(x2), int(y2)],
        })

    # draw
    img_with = draw_boxes_on_image(img, detections)
    return detections, _pil_to_bytes(img_with)
