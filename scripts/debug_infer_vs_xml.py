# scripts/debug_infer_vs_xml.py
import sys, os
# ensure project root is on sys.path so `src` imports work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import torch

# adjust paths if needed
from src.proposals import get_selective_search_proposals, filter_proposals
from src.warp_and_crop import crop_and_resize_patches
from src.model import build_model, LABELS
# try to import box_iou from train; if not available, define a local version
try:
    from src.train import box_iou  # reuse same IoU fn if available
except Exception:
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
        if areaA + areaB - inter <= 0:
            return 0.0
        return inter / float(areaA + areaB - inter + 1e-8)

# default file paths (edit if your files are named differently)
IMG_PATH = "data/images/sample.jpg"
XML_PATH = "data/annotations/sample.xml"
CKPT = "checkpoints/rcnn_head.pth"

def parse_voc(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = []
    for o in root.findall("object"):
        name = o.find("name").text
        b = o.find("bndbox")
        xmin = int(float(b.find("xmin").text))
        ymin = int(float(b.find("ymin").text))
        xmax = int(float(b.find("xmax").text))
        ymax = int(float(b.find("ymax").text))
        objs.append({"label": name, "bbox": (xmin,ymin,xmax,ymax)})
    return objs

def print_gt():
    objs = parse_voc(XML_PATH)
    print("Ground-truth objects from", XML_PATH)
    for i,o in enumerate(objs,1):
        print(f" GT[{i}] label={o['label']} bbox={o['bbox']}")

def proposals_and_iou(max_proposals=1000, show_top=30):
    pil = Image.open(IMG_PATH).convert("RGB")
    raw = get_selective_search_proposals(pil, mode="fast")
    boxes = filter_proposals(raw, min_area=500, max_proposals=max_proposals)
    print("\nTotal proposals after filtering:", len(boxes))
    gts = parse_voc(XML_PATH)
    # for each proposal, compute best IoU and which GT it matches
    rows = []
    for j, b in enumerate(boxes[:max_proposals]):  # consider up to max_proposals
        best_iou = 0.0
        best_label = None
        for g in gts:
            iou = box_iou(b, g["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_label = g["label"]
        rows.append((j, b, best_iou, best_label))
    # sort by best_iou desc and print top entries that overlap GTs
    rows = sorted(rows, key=lambda x: x[2], reverse=True)
    print(f"\nTop proposals by IoU with GT (showing top {show_top}):")
    printed = 0
    for r in rows:
        if r[2] <= 0.0:
            continue
        print(" P#%03d  box=%s  IoU=%.3f  matched_gt=%s" % (r[0], r[1], r[2], str(r[3])))
        printed += 1
        if printed >= show_top:
            break
    if printed == 0:
        print(" (No proposals with IoU > 0 found in the first", max_proposals, "proposals.)")
    return pil, boxes

def model_preds_on_top_proposals(pil, boxes, top_k=200):
    # crop proposals (use first top_k proposals)
    top_boxes = boxes[:top_k]
    patches, mapped = crop_and_resize_patches(pil, top_boxes)
    if patches is None or patches.shape[0]==0:
        print("No patches from crop_and_resize_patches")
        return
    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(num_classes=len(LABELS), device=str(device))
    if os.path.exists(CKPT):
        try:
            model.load_state_dict(torch.load(CKPT, map_location="cpu"))
            print("\nLoaded checkpoint:", CKPT)
        except Exception as e:
            print("\nFailed loading checkpoint (will continue with random weights):", e)
    else:
        print("\nNo checkpoint found at", CKPT, " — predictions will be from random weights")
    model.to(device)
    model.eval()
    # predict batch (try to reuse predict_batch if available)
    try:
        from src.model import predict_batch
        probs = predict_batch(model, patches, device, batch_size=32)  # returns N x C
    except Exception:
        # fallback: run model forward and softmax
        with torch.no_grad():
            patches = patches.to(device)
            logits = model(patches)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

    # ensure numpy
    if isinstance(probs, torch.Tensor):
        probs_np = probs.detach().cpu().numpy()
    else:
        probs_np = np.array(probs)
    print("\nModel predictions on proposals (showing top class per proposal):")
    for i in range(min(50, probs_np.shape[0])):
        row = probs_np[i]
        # ignore background col 0; find best foreground
        fg = row[1:len(LABELS)]
        best_rel = int(np.argmax(fg))
        best_score = float(fg[best_rel])
        best_idx = best_rel + 1
        print(f" prop#{i:03d} box={mapped[i]}  best_class_idx={best_idx} name={LABELS[best_idx]} score={best_score:.3f}")

if __name__ == "__main__":
    if not os.path.exists(IMG_PATH) or not os.path.exists(XML_PATH):
        print("Place image at", IMG_PATH, "and xml at", XML_PATH)
        sys.exit(1)
    print_gt()
    pil, boxes = proposals_and_iou()
    model_preds_on_top_proposals(pil, boxes)
