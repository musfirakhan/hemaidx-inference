# HemAIDX

Deployment of **HemAIDX**, a portable, low-cost system for automated blood smear screening that combines **virtual staining** with **white blood cell detection**. 

This repository contains the backend service that runs the trained models on new microscope images. The models were deployed both on a web platform and on a portable device (Raspberry Pi 5 with a camera module mounted on a microscope).


### API (`app.py`)
A FastAPI service with two endpoints:

| Endpoint | Input | Output |
|---|---|---|
| `POST /virtual-staining/` | unstained smear image | virtually stained image |
| `POST /classify/` | smear image | image with WBC boxes and labels |

## Setup

```bash
pip install -r requirements.txt
```

Place the ONNX models in `models/` (not included in this repository), or point to them with environment variables:

| Variable | Default |
|---|---|
| `HEMAIDX_STAIN_MODEL` | `models/pix2pixhd_generator.onnx` |
| `HEMAIDX_DETECT_MODEL` | `models/rtdetr_wbc.onnx` |
| `HEMAIDX_OUTPUT_DIR` | `outputs` |
| `HEMAIDX_FRONTEND_ORIGIN` | `http://localhost:5173` |

Run the service:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```


## Team
HemAIDX was developed by a team at the Department of Computer & Software Engineering, NUST, supervised by Prof. Muhammad Usman Akram, in collaboration with clinical partners.

Trained model weights and clinical images not included.
