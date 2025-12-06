import os
import random
import xml.etree.ElementTree as ET
from typing import List, Dict

from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from src.proposals import get_selective_search_proposals, filter_proposals
from src.warp_and_crop import crop_and_resize_patches
from src.model import build_model, LABELS

# ================= CONFIG & PATHS =================

DATA_ROOT = "data"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
ANN_DIR = os.path.join(DATA_ROOT, "annotations")
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

NUM_CLASSES = len(LABELS)  # background + foreground classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_PROPOSALS_PER_IMAGE = 150
SAMPLES_PER_IMAGE = 32      # how many proposals to sample per image per epoch
POS_RATIO = 0.25            # desired fraction of positives
BATCH_SIZE = 64
EPOCHS = 3
LR = 1e-4
WEIGHT_DECAY = 1e-4
PRINT_EVERY = 50

# ================= TRANSFORMS =====================

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================= XML PARSING ====================

def parse_voc_xml(xml_path: str) -> List[Dict]:
    """
    Returns list of dicts: {'label': label_name, 'bbox': (x1,y1,x2,y2)}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bnd = obj.find("bndbox")
        x1 = int(float(bnd.find("xmin").text))
        y1 = int(float(bnd.find("ymin").text))
        x2 = int(float(bnd.find("xmax").text))
        y2 = int(float(bnd.find("ymax").text))
        objs.append({"label": name, "bbox": (x1, y1, x2, y2)})
    return objs

def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    if areaA + areaB - inter <= 0:
        return 0.0
    return inter / float(areaA + areaB - inter + 1e-8)

# Map label names to class indices (must match LABELS)
LABEL_TO_IDX = {name: idx for idx, name in enumerate(LABELS)}  # background included

# ============= DATASET: PROPOSALS -> PATCHES =====

class RCNNProposalDataset(Dataset):
    def __init__(self, images_dir, ann_dir,
                 max_proposals=MAX_PROPOSALS_PER_IMAGE,
                 samples_per_image=SAMPLES_PER_IMAGE):
        self.images_dir = images_dir
        self.ann_dir = ann_dir
        self.samples_per_image = samples_per_image
        self.max_proposals = max_proposals

        # collect image ids (files with annotations)
        files = []
        for fname in os.listdir(self.ann_dir):
            if not fname.endswith(".xml"):
                continue
            img_id = os.path.splitext(fname)[0]
            img_path = os.path.join(self.images_dir, img_id + ".jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.images_dir, img_id + ".png")
                if not os.path.exists(img_path):
                    continue
            files.append(img_id)
        self.ids = files
        if len(self.ids) == 0:
            raise RuntimeError("No annotation files found in " + self.ann_dir)

    def __len__(self):
        # treat each image as one sample in the dataset
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        # load image and annotations
        xml_path = os.path.join(self.ann_dir, img_id + ".xml")
        annos = parse_voc_xml(xml_path)
        img_path = os.path.join(self.images_dir, img_id + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.images_dir, img_id + ".png")
        pil = Image.open(img_path).convert("RGB")

        # proposals
        raw_boxes = get_selective_search_proposals(pil, mode="fast")
        boxes = filter_proposals(raw_boxes, min_area=500, max_proposals=self.max_proposals)
        if len(boxes) == 0:
            boxes = [(0, 0, pil.width, pil.height)]

        # compute IoU of each proposal with all GT boxes, find best
        proposal_labels = []   # (box, label_idx, max_iou)
        for b in boxes:
            best_iou = 0.0
            best_label = None
            for obj in annos:
                iou = box_iou(b, obj["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_label = obj["label"]
            if best_iou >= 0.5 and best_label in LABEL_TO_IDX:
                label_idx = LABEL_TO_IDX[best_label]
            elif best_iou <= 0.3:
                label_idx = LABEL_TO_IDX["background"]
            else:
                label_idx = None
            proposal_labels.append((b, label_idx, best_iou))

        # sample positives and negatives
        positives = [p for p in proposal_labels if p[1] is not None and p[1] != LABEL_TO_IDX["background"]]
        negatives = [p for p in proposal_labels if p[1] == LABEL_TO_IDX["background"]]

        n_pos = int(self.samples_per_image * POS_RATIO)
        n_neg = self.samples_per_image - n_pos

        sampled = []
        if len(positives) > 0:
            sampled_pos = random.sample(positives, min(len(positives), n_pos))
            sampled.extend(sampled_pos)

        if len(sampled) < n_pos and len(negatives) > 0:
            needed = n_pos - len(sampled)
            sampled.extend(random.sample(negatives, min(len(negatives), needed)))

        if len(negatives) > 0:
            sampled_neg = random.sample(negatives, min(len(negatives), n_neg))
            sampled.extend(sampled_neg)

        if len(sampled) < self.samples_per_image:
            others = [p for p in proposal_labels if p not in sampled and p[1] is not None]
            if others:
                sampled.extend(random.sample(others, min(len(others), self.samples_per_image - len(sampled))))

        boxes_for_crop = [p[0] for p in sampled]
        labels = [p[1] if p[1] is not None else LABEL_TO_IDX["background"] for p in sampled]
        patches_tensor, _ = crop_and_resize_patches(pil, boxes_for_crop)

        if patches_tensor.shape[0] != len(labels):
            labels = labels[:patches_tensor.shape[0]]

        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return patches_tensor, labels_tensor

# ============= COLLATE FUNCTION ===================

def rcnn_collate(batch):
    patches = [b[0] for b in batch if b[0] is not None and b[0].shape[0] > 0]
    labels = [b[1] for b in batch if b[1] is not None and b[1].shape[0] > 0]
    if len(patches) == 0:
        return torch.empty((0, 3, 224, 224)), torch.empty((0,), dtype=torch.long)
    X = torch.cat(patches, dim=0)
    Y = torch.cat(labels, dim=0)
    return X, Y

# ============= TRAIN LOOP =========================

def train():
    print("Device:", DEVICE)
    dataset = RCNNProposalDataset(IMAGES_DIR, ANN_DIR,
                                  max_proposals=MAX_PROPOSALS_PER_IMAGE,
                                  samples_per_image=SAMPLES_PER_IMAGE)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            collate_fn=rcnn_collate, num_workers=2)

    # build model and freeze backbone layers (except final head)
    model, _ = build_model(num_classes=NUM_CLASSES, device=DEVICE.type)
    model.to(DEVICE)

    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=LR, weight_decay=WEIGHT_DECAY)

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for i, (x_batch, y_batch) in enumerate(dataloader):
            if x_batch.shape[0] == 0:
                continue
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            _, preds = torch.max(logits, dim=1)
            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()
            global_step += 1

            if global_step % PRINT_EVERY == 0:
                avg_loss = running_loss / total if total > 0 else 0.0
                acc = correct / total if total > 0 else 0.0
                print(f"[E{epoch+1}] Step {global_step} avg_loss={avg_loss:.4f} acc={acc:.4f}")

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}/{EPOCHS} finished. loss={epoch_loss:.4f} acc={epoch_acc:.4f}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, "rcnn_head.pth")
        torch.save(model.state_dict(), ckpt_path)
        print("Saved checkpoint to", ckpt_path)

    print("Training complete.")

if __name__ == "__main__":
    train()
