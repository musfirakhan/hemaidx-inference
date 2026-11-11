"""White blood cell detection with an RT-DETR model exported to ONNX.

Raw detections are filtered by confidence and then de-duplicated with
cross-class non-maximum suppression (NMS), since the same cell can otherwise
be reported twice under different class labels.
"""

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms
from torchvision.transforms import ToTensor

CLASSES = ["Others", "lymphocyte", "monocyte", "neutrophil"]
INPUT_SIZE = 640
SCORE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.4


def detect_cells(image, session):
    """Return (boxes, scores, labels) for one image, after post-processing."""
    resized = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    tensor = ToTensor()(resized)[None]
    size = torch.tensor([[INPUT_SIZE, INPUT_SIZE]])

    labels, boxes, scores = session.run(
        output_names=None,
        input_feed={"images": tensor.numpy(), "orig_target_sizes": size.numpy()},
    )
    labels = torch.tensor(labels[0])
    boxes = torch.tensor(boxes[0])
    scores = torch.tensor(scores[0])

    # 1. Confidence filtering
    keep = scores > SCORE_THRESHOLD
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    # 2. Cross-class NMS
    keep = nms(boxes, scores, NMS_IOU_THRESHOLD)
    return boxes[keep], scores[keep], labels[keep]


def draw_detections(image, boxes, labels):
    """Draw boxes and class names, returned at the original image size."""
    width, height = image.size
    canvas = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for box, label in zip(boxes, labels):
        draw.rectangle(box.tolist(), outline="red", width=2)
        draw.text((box[0], box[1] - 10), CLASSES[int(label.item())], fill="blue", font=font)

    return canvas.resize((width, height))
