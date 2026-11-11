"""FastAPI service for HemAIDX: virtual staining and white blood cell detection.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os

import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from hemaidx.detection import detect_cells, draw_detections
from hemaidx.staining import virtually_stain

STAIN_MODEL_PATH = os.getenv("HEMAIDX_STAIN_MODEL", "models/pix2pixhd_generator.onnx")
DETECT_MODEL_PATH = os.getenv("HEMAIDX_DETECT_MODEL", "models/rtdetr_wbc.onnx")
OUTPUT_DIR = os.getenv("HEMAIDX_OUTPUT_DIR", "outputs")
FRONTEND_ORIGIN = os.getenv("HEMAIDX_FRONTEND_ORIGIN", "http://localhost:5173")

os.makedirs(OUTPUT_DIR, exist_ok=True)
Image.MAX_IMAGE_PIXELS = 1_866_240_000  # whole-slide-sized microscope images
ort.set_default_logger_severity(3)

# Use the GPU when available, otherwise fall back to CPU
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
stain_session = ort.InferenceSession(STAIN_MODEL_PATH, providers=PROVIDERS)
detect_session = ort.InferenceSession(DETECT_MODEL_PATH, providers=PROVIDERS)

app = FastAPI(title="HemAIDX Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def save_upload(file: UploadFile) -> str:
    path = os.path.join(OUTPUT_DIR, os.path.basename(file.filename))
    with open(path, "wb") as f:
        f.write(file.file.read())
    return path


@app.post("/virtual-staining/")
async def virtual_staining(file: UploadFile = File(...)):
    try:
        image_path = save_upload(file)
        stained = virtually_stain(image_path, stain_session, OUTPUT_DIR)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        result_path = os.path.join(OUTPUT_DIR, f"{base_name}_stained.jpg")
        stained.save(result_path)
        return FileResponse(result_path, media_type="image/jpeg", filename=f"stained_{file.filename}")
    except Exception as e:
        return {"error": f"Virtual staining failed: {e}"}


@app.post("/classify/")
async def classify(file: UploadFile = File(...)):
    try:
        image_path = save_upload(file)
        image = Image.open(image_path)
        boxes, _, labels = detect_cells(image, detect_session)
        annotated = draw_detections(image, boxes, labels)

        result_path = os.path.join(OUTPUT_DIR, f"detected_{os.path.basename(file.filename)}")
        annotated.save(result_path)
        return FileResponse(result_path, media_type="image/jpeg", filename=f"detected_{file.filename}")
    except Exception as e:
        return {"error": f"Detection failed: {e}"}
