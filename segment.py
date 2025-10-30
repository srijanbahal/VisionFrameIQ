import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ------------------------------------
# Force CPU mode globally
# ------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# ------------------------------------
# Load CLIPSeg (for later text-based region guidance)
# ------------------------------------
print("Loading CLIPSeg model...")
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
    "CIDAS/clipseg-rd64-refined"
).to(DEVICE)
print("✅ CLIPSeg model loaded successfully.")

# ------------------------------------
# Load SAM 2.1 manually on CPU
# ------------------------------------
print("Loading SAM 2.1 model (CPU mode)...")
sam2_model = build_sam2_hf("facebook/sam2-hiera-base-plus", device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)
print("✅ SAM 2.1 model loaded successfully on CPU.")

# ------------------------------------
# Region Highlighting Function
# ------------------------------------
def highlight_region_with_sam2(image_path: str, mask_prompts=None):
    """
    Highlights object regions using SAM 2.1.
    - If mask_prompts is None → automatic segmentation mode.
    - If mask_prompts is provided (dict with 'point_coords' and 'point_labels') → prompt-based.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    with torch.inference_mode():
        predictor.set_image(img)
        
        # Auto segmentation mode (no prompts)
        if mask_prompts is None:
            masks, scores, _ = predictor.predict(None)
        else:
            # Expect properly formatted dict with coords and labels
            masks, scores, _ = predictor.predict(**mask_prompts)

        if len(masks) == 0:
            print("⚠️ No masks generated for this image.")
            return Image.fromarray(img)

        combined = np.zeros_like(img)
        for m in masks:
            mask3 = np.repeat(m[..., None], 3, axis=-1)
            color = np.random.randint(0, 255, size=(1, 1, 3), dtype=np.uint8)
            combined = np.where(mask3, color, combined)

        overlay = cv2.addWeighted(img, 0.6, combined, 0.4, 0)
    return Image.fromarray(overlay)


# ------------------------------------
# Test mode
# ------------------------------------
if __name__ == "__main__":
    test_image = "0200182002.jpg"  # put any small jpg here
    result = highlight_region_with_sam2(test_image)
    result.show()


