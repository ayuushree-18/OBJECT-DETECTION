from fastapi.responses import RedirectResponse
# web/app.py
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from src.faster_infer import predict_faster_from_bytes
from src.model import build_model
from src.infer import predict_from_bytes

app = FastAPI()
app.mount("/static", StaticFiles(directory="web/static"), name="static")

MODEL = None
DEVICE = None

@app.on_event("startup")
@app.on_event("startup")

def load_model():
    global MODEL, DEVICE
    MODEL, DEVICE = build_model(num_classes=4, device="cpu")
    ckpt = "checkpoints/rcnn_head.pth"
    import os, torch
    if os.path.exists(ckpt):
        MODEL.load_state_dict(torch.load(ckpt, map_location="cpu"))
        MODEL.eval()
        print("Loaded checkpoint:", ckpt)
    else:
        print("No checkpoint found at", ckpt, "- running with untrained head.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    detections, img_bytes = predict_faster_from_bytes(contents, score_thresh=0.6)
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return JSONResponse({"detections": detections, "image_base64": img_b64})

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")
