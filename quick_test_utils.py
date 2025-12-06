from PIL import Image
from src.utils import nms, draw_boxes_on_image

if __name__ == "__main__":
    # Sample dummy boxes
    boxes = [
        (50, 50, 200, 200),
        (60, 60, 210, 210),  # overlaps highly with box 0
        (300, 80, 420, 200)  # separate box
    ]
    scores = [0.9, 0.85, 0.92]

    keep = nms(boxes, scores, iou_thresh=0.4)
    print("Boxes kept after NMS:", keep)

    # Draw boxes on sample.jpg for visualization
    img = Image.open("data/sample.jpg").convert("RGB")
    detections = [
        {"label": "car", "score": 0.9,  "box": boxes[0]},
        {"label": "car", "score": 0.85, "box": boxes[1]},
        {"label": "person", "score": 0.92, "box": boxes[2]},
    ]
    out = draw_boxes_on_image(img, detections)
    out.save("data/test_utils_output.png")
    print("Saved drawn image at data/test_utils_output.png")
