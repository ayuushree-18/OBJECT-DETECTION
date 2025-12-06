# src/infer.py
from typing import List, Dict, Tuple
from PIL import Image
import io

import numpy as np
import torch

from src.proposals import get_selective_search_proposals, filter_proposals
from src.warp_and_crop import crop_and_resize_patches
from src.model import predict_batch, LABELS
from src.utils import nms, draw_boxes_on_image

# default per-class thresholds (can be overridden when calling predict_from_bytes)
DEFAULT_THRESH = {"person": 0.1, "bicycle": 0.1, "car": 0.1}

def _ensure_numpy(probs):
    """Convert torch tensor to numpy array if needed."""
    if probs is None:
        return None
    if isinstance(probs, torch.Tensor):
        return probs.detach().cpu().numpy()
    return np.array(probs)

def _boxes_scores_labels_from_probs(boxes: List[Tuple[int,int,int,int]], probs, thresholds: Dict[str,float]):
    """
    boxes: list of (x1,y1,x2,y2) matching rows of probs
    probs: numpy array (N, C) where column index matches LABELS order
    returns:
      per_class dict: class_name -> list of (box, score)
    Behavior:
      - For each proposal choose the single best foreground class (argmax over classes excluding background index 0).
      - Accept that class only if score >= thresholds[class_name].
    """
    per_class = {c: [] for c in thresholds.keys()}
    probs = _ensure_numpy(probs)
    if probs is None or probs.shape[0] == 0:
        return per_class
    # sanity: probs shape should have C columns >= len(LABELS)
    C = probs.shape[1]
    expected_C = len(LABELS)
    if C < expected_C:
        # fallback: if model returned fewer classes, we can't reliably map; return empty
        return per_class

    # For each proposal: choose best class among LABELS[1:]
    for i, p in enumerate(probs):
        # skip background col 0; find best foreground index
        fg_scores = p[1:expected_C]  # length = expected_C-1
        best_rel_idx = int(np.argmax(fg_scores))   # 0..(C-2)
        best_score = float(fg_scores[best_rel_idx])
        best_cls_index = best_rel_idx + 1  # map back to full LABELS index
        best_cls_name = LABELS[best_cls_index]
        required_thresh = thresholds.get(best_cls_name, 1e9)  # if no threshold defined, require very high (skip)
        if best_score >= required_thresh:
            per_class[best_cls_name].append((boxes[i], best_score))
    return per_class

def pil_image_to_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def predict_from_bytes(image_bytes: bytes,
                       model,
                       device,
                       thresholds: Dict[str,float]=None,
                       max_proposals: int=1000,
                       top_k: int=50):
    """
    Full R-CNN inference on one image.
    Returns:
      detections: list of {"label","score","box":[x1,y1,x2,y2]}
      image_with_boxes_bytes: PNG bytes
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESH

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 1) proposals
    raw_boxes = get_selective_search_proposals(pil_image, mode="fast")
    boxes = filter_proposals(raw_boxes, min_area=500, max_proposals=max_proposals)
    if len(boxes) == 0:
        # nothing found: return original image
        return [], pil_image_to_bytes(pil_image)

    # 2) crop and resize
    patches_tensor, mapped_boxes = crop_and_resize_patches(pil_image, boxes)
    if patches_tensor is None or patches_tensor.shape[0] == 0:
        return [], pil_image_to_bytes(pil_image)

    # 3) predict probabilities (N x C)
    probs = predict_batch(model, patches_tensor, device, batch_size=32)  # may be torch.Tensor or np.array

    # 4) per-class thresholding (single best class per proposal)
    per_class = _boxes_scores_labels_from_probs(mapped_boxes, probs, thresholds)

    # 5) NMS and collect detections
    detections = []
    for cls_name, items in per_class.items():
        if not items:
            continue
        cls_boxes = [it[0] for it in items]
        cls_scores = [it[1] for it in items]
        keep_idx = nms(cls_boxes, cls_scores, iou_thresh=0.4)
        for ki in keep_idx:
            detections.append({"label": cls_name, "score": float(cls_scores[ki]), "box": cls_boxes[ki]})

    # 6) fallback: if no detections, optionally include top proposals (best scoring foreground) to ensure visible result
    if len(detections) == 0:
        probs_np = _ensure_numpy(probs)
        if probs_np is not None and probs_np.shape[0] > 0:
            # find highest foreground score across all proposals
            expected_C = len(LABELS)
            if probs_np.shape[1] >= expected_C:
                fg_scores = probs_np[:, 1:expected_C]
                best_flat = int(np.argmax(fg_scores))
                best_prop_idx = best_flat // fg_scores.shape[1]
                best_cls_rel = best_flat % fg_scores.shape[1]
                best_cls_index = best_cls_rel + 1
                best_cls_name = LABELS[best_cls_index]
                best_score = float(fg_scores[best_prop_idx, best_cls_rel])
                detections.append({"label": best_cls_name, "score": best_score, "box": mapped_boxes[best_prop_idx]})

    # 7) sort & limit
    detections = sorted(detections, key=lambda x: x["score"], reverse=True)[:top_k]

    # 8) draw
    img_with = draw_boxes_on_image(pil_image, detections)
    return detections, pil_image_to_bytes(img_with)
