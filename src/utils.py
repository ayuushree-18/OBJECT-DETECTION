from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    iou = inter / float(areaA + areaB - inter + 1e-8)
    return iou

def nms(boxes: List[Tuple[int,int,int,int]], scores: List[float], iou_thresh: float=0.4):
    """
    boxes: list of (x1,y1,x2,y2); scores: list of floats
    Returns indices of boxes to keep.
    """
    if len(boxes) == 0:
        return []
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(int(i))
        rem = []
        for j in idxs[1:]:
            if box_iou(boxes[i], boxes[j]) > iou_thresh:
                continue
            rem.append(j)
        idxs = np.array(rem, dtype=int)
    return keep

def draw_boxes_on_image(pil_image: Image.Image, detections: List[Dict], class_colors: Dict[str,Tuple[int,int,int]]=None):
    img = pil_image.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for d in detections:
        box = d["box"]
        label = d.get("label", "")
        score = d.get("score", None)
        color = (255, 0, 0)
        if class_colors and label in class_colors:
            color = class_colors[label]
        draw.rectangle(box, outline=color, width=2)
        text = f"{label}"
        if score is not None:
            text = f"{label}: {score:.2f}"
        if font:
            draw.text((box[0] + 3, box[1] + 3), text, fill=color, font=font)
        else:
            draw.text((box[0] + 3, box[1] + 3), text, fill=color)
    return img
