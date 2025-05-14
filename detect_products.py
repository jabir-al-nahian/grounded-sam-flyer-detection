import cv2
import torch
from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor
import numpy as np

# Load Grounding DINO config and model
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./models/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT_PATH = "./models/sam_vit_b_01ec64.pth"
image_path ="./images/flyer.jpg"

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
gdino_model = load_model(GROUNDING_DINO_CONFIG_PATH, CHECKPOINT_PATH)
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=device)
sam_predictor = SamPredictor(sam)

# Load image
image_path = "./images/flyer.jpg"
image_source, image = load_image(image_path)

# Predict using Grounding DINO
TEXT_PROMPT = "product"  # Can be replaced with specific items
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

boxes, logits, phrases = predict(
    model=gdino_model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=device
)

# Segment using SAM
sam_predictor.set_image(image_source)
transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, image_source.shape[:2])
masks, _, _ = sam_predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.imshow(image_source)
for mask in masks:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
for box, label in zip(boxes, phrases):
    show_box(box.numpy(), plt.gca(), label)
plt.axis("off")
plt.show()
