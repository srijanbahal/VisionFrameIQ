import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Load CLIPSeg ---
print("Loading CLIPSeg model...")
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
    "CIDAS/clipseg-rd64-refined"
).to(DEVICE)
print("CLIPSeg model loaded.")

# --- Load SAM 2 ---
print("Loading SAM 2 model from Hugging Face...")
print("Loading SAM 2 model...")
sam2_model = build_sam2("facebook/sam2-hiera-base-plus", device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)
print("SAM 2 model loaded.")

def highlight_region_with_sam2(image_path: str, mask_prompts=None):
    """
    Generate automatic or prompted mask highlighting using SAM 2.1.
    mask_prompts: can be None, or list of dicts with point_coords / boxes / etc.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        predictor.set_image(img)
        # If no manual prompts, let SAM auto-segment
        masks, scores, _ = predictor.predict(mask_prompts)
        if len(masks) == 0:
            return Image.fromarray(img)
        # Combine multiple masks into one overlay
        combined = np.zeros_like(img)
        for m in masks:
            mask3 = np.repeat(m[..., None], 3, axis=-1)
            color = np.random.randint(0, 255, size=(1, 1, 3), dtype=np.uint8)
            combined = np.where(mask3, color, combined)
        overlay = cv2.addWeighted(img, 0.6, combined, 0.4, 0)
    return Image.fromarray(overlay)