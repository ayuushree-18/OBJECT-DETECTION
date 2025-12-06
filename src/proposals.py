from typing import List, Tuple
import numpy as np
from PIL import Image

def _to_xyxy(box):
    x, y, w, h = box
    return (int(x), int(y), int(x + w), int(y + h))

def get_selective_search_proposals(image, mode: str = "fast") -> List[Tuple[int,int,int,int]]:
    """
    image: PIL.Image or numpy array (H,W,3)
    mode: 'fast' or 'quality'
    returns: list of boxes as (x1,y1,x2,y2)
    """
    try:
        import selectivesearch
    except Exception:
        selectivesearch = None

    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.array(image)

    boxes_xyxy = []
    if selectivesearch is not None:
        img_lbl, regions = selectivesearch.selective_search(img, scale=500 if mode == "fast" else 1500,
                                                            sigma=0.9, min_size=10)
        seen = set()
        for r in regions:
            if r.get("rect") is None:
                continue
            x, y, w, h = r["rect"]
            if w == 0 or h == 0:
                continue
            key = (x, y, w, h)
            if key in seen:
                continue
            seen.add(key)
            boxes_xyxy.append(_to_xyxy((x, y, w, h)))
    else:
        # Fallback: try OpenCV's selective search from opencv-contrib
        try:
            import cv2
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            if isinstance(image, np.ndarray):
                ss.setBaseImage(image)
            else:
                ss.setBaseImage(np.array(image))
            if mode == "fast":
                ss.switchToSelectiveSearchFast()
            else:
                ss.switchToSelectiveSearchQuality()
            rects = ss.process()
            for r in rects:
                x, y, w, h = r
                boxes_xyxy.append(_to_xyxy((x, y, w, h)))
        except Exception as e:
            raise RuntimeError("selectivesearch not installed and opencv-contrib fallback failed: " + str(e))

    return boxes_xyxy

def filter_proposals(boxes: List[Tuple[int,int,int,int]],
                     min_area: int = 500,
                     aspect_ratio_thresh: float = 4.0,
                     max_proposals: int = 1000) -> List[Tuple[int,int,int,int]]:
    """
    boxes: list of (x1,y1,x2,y2)
    Filters out tiny boxes, extreme aspect ratios. Keeps up to max_proposals (first ones).
    """
    filtered = []
    for (x1,y1,x2,y2) in boxes:
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if w * h < min_area:
            continue
        if h == 0 or w == 0:
            continue
        ar = max(w/h, h/w)
        if ar > aspect_ratio_thresh:
            continue
        filtered.append((x1,y1,x2,y2))
        if len(filtered) >= max_proposals:
            break
    return filtered
