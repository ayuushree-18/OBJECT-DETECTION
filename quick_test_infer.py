from src.model import build_model
from src.infer import predict_from_bytes
import sys

if __name__ == "__main__":
    model, device = build_model(device="cpu")
    with open("data/sample.jpg","rb") as f:
        img_bytes = f.read()
    dets, img_bytes_out = predict_from_bytes(img_bytes, model, device, thresholds={"person":0.5,"bicycle":0.5,"car":0.5}, max_proposals=300)
    print("Detections:", dets[:10])
    with open("data/test_infer_out.png","wb") as f:
        f.write(img_bytes_out)
    print("Saved output with boxes to data/test_infer_out.png")
