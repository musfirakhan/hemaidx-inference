"""Patch-based virtual staining with a Pix2PixHD generator exported to ONNX.

Microscope images are much larger than the generator's input, so the image is
split into overlapping 512x512 patches, each patch is stained independently,
and the patches are blended back together with a Gaussian weight mask to
avoid visible seams.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rembg
from PIL import Image, ImageOps

PATCH_SIZE = 512
BACKGROUND_RATIO_THRESHOLD = 0.4


def clear_folder(folder_path):
    """Delete all files in a folder (keeps the folder itself)."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)


def create_patches(image_path, patch_size, output_dir):
    """Split an image into patches with 50% overlap.

    The image is padded so its size is divisible by the stride (patch_size / 2).
    Returns the patch folder, the padded size and the padding that was added.
    """
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    patch_dir = os.path.join(output_dir, f"patches_{base_name}")
    os.makedirs(patch_dir, exist_ok=True)
    clear_folder(patch_dir)

    stride = patch_size // 2
    pad_x = (stride - (orig_w % stride)) % stride
    pad_y = (stride - (orig_h % stride)) % stride

    padded = ImageOps.expand(image, (0, 0, pad_x, pad_y), fill=0)
    pad_w, pad_h = padded.size

    for y in range(0, pad_h - patch_size + 1, stride):
        for x in range(0, pad_w - patch_size + 1, stride):
            patch = padded.crop((x, y, x + patch_size, y + patch_size))
            patch.save(os.path.join(patch_dir, f"patch_{x:04d}_{y:04d}.png"))

    return patch_dir, (pad_w, pad_h), (pad_x, pad_y)


def stain_patch(patch_file, session, patch_dir, processed_dir):
    """Run one patch through the generator and save the stained result."""
    x = int(patch_file.split("_")[1])
    y = int(patch_file.split("_")[2].split(".")[0])

    img = Image.open(os.path.join(patch_dir, patch_file)).convert("RGB")
    # Scale to [-1, 1], HWC -> NCHW
    tensor = (np.asarray(img, dtype=np.float32) / 127.5) - 1.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]

    result = session.run(None, {"input": tensor})[0]

    # Back to [0, 255], NCHW -> HWC
    result = (np.transpose(result[0], (1, 2, 0)) + 1.0) * 127.5
    result = np.clip(result, 0, 255).astype(np.uint8)
    Image.fromarray(result).save(
        os.path.join(processed_dir, f"processed_{x:04d}_{y:04d}.png")
    )


def stain_patches(session, patch_dir, processed_dir, max_workers=12):
    """Stain all patches in parallel."""
    os.makedirs(processed_dir, exist_ok=True)
    clear_folder(processed_dir)

    patch_files = [f for f in os.listdir(patch_dir) if f.endswith(".png")]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(
            lambda f: stain_patch(f, session, patch_dir, processed_dir),
            patch_files,
        ))


def reconstruct_image(processed_dir, padded_size, patch_size, pad):
    """Blend overlapping stained patches back into one image.

    Each patch is weighted by a 2D Gaussian mask (highest at the centre), so
    overlapping regions transition smoothly instead of showing tile seams.
    """
    pad_w, pad_h = padded_size
    pad_x, pad_y = pad
    canvas = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    weights = np.zeros((pad_h, pad_w), dtype=np.float32)

    gx, gy = np.meshgrid(np.linspace(-1, 1, patch_size), np.linspace(-1, 1, patch_size))
    mask = np.exp(-(gx ** 2 + gy ** 2))
    mask /= mask.max()

    for patch_file in os.listdir(processed_dir):
        if not patch_file.startswith("processed_"):
            continue
        parts = patch_file.split("_")
        x = int(parts[1])
        y = int(parts[2].split(".")[0])

        patch = np.asarray(Image.open(os.path.join(processed_dir, patch_file)))
        y_end, x_end = min(y + patch_size, pad_h), min(x + patch_size, pad_w)
        h, w = y_end - y, x_end - x

        canvas[y:y_end, x:x_end] += patch[:h, :w] * mask[:h, :w, np.newaxis]
        weights[y:y_end, x:x_end] += mask[:h, :w]

    canvas = np.divide(
        canvas, weights[..., np.newaxis],
        out=np.zeros_like(canvas), where=weights[..., np.newaxis] > 0,
    )
    final = canvas[: pad_h - pad_y, : pad_w - pad_x]
    return Image.fromarray(final.astype(np.uint8))


def restore_background(original, stained):
    """Keep the original background where the field of view is mostly empty.

    rembg separates the foreground (the smear) from the background. If more
    than 40% of the image is background, the original background is composited
    over the stained output so the generator does not hallucinate colour there.
    """
    original_rgba = original.convert("RGBA")
    alpha = Image.fromarray(rembg.remove(np.array(original_rgba))).getchannel("A")

    binary = np.asarray(alpha.point(lambda v: 255 if v > 128 else 0))
    background_ratio = (binary / 255).sum() / binary.size

    if background_ratio <= BACKGROUND_RATIO_THRESHOLD:
        return stained.convert("RGB")

    background_mask = alpha.point(lambda v: 255 - v)
    background = Image.composite(
        original_rgba, Image.new("RGBA", original_rgba.size, (0, 0, 0, 0)), background_mask
    )
    combined = Image.alpha_composite(
        stained.convert("RGBA").resize(original_rgba.size), background
    )
    return combined.convert("RGB")


def virtually_stain(image_path, session, output_dir, patch_size=PATCH_SIZE):
    """Full pipeline: patch -> stain -> blend -> background restore."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    patch_dir, padded_size, pad = create_patches(image_path, patch_size, output_dir)
    processed_dir = os.path.join(output_dir, f"processed_{base_name}")

    stain_patches(session, patch_dir, processed_dir)
    reconstructed = reconstruct_image(processed_dir, padded_size, patch_size, pad)

    return restore_background(Image.open(image_path), reconstructed)
