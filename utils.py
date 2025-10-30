import cv2, os, torch, numpy as np
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration,
)
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid
from segment_anything import sam_model_registry, SamPredictor
import torch.nn.functional as F



# --- Load models once ---
print("Loading models...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP model loaded.")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP processor loaded.")

caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
print("BLIP caption model loaded.")
caption_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
print("BLIP caption processor loaded.")


# Extract frames per video into frames/<video_name>/
def extract_frames(video_path, output_folder="frames", fps=2):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval = int(frame_rate // fps) if frame_rate else 1
    i, count = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % interval == 0:
            frame_path = os.path.join(video_folder, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            count += 1
        i += 1

    cap.release()
    return [
        os.path.join(video_folder, f) for f in sorted(os.listdir(video_folder))
    ], video_name


def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return emb.squeeze().cpu().numpy()


def smooth_embeddings(embeddings, window=3):
    smoothed = []
    for i in range(len(embeddings)):
        start = max(0, i - window)
        end = min(len(embeddings), i + window + 1)
        avg_emb = np.mean(embeddings[start:end], axis=0)
        smoothed.append(avg_emb)
    return smoothed


def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = caption_processor(image, return_tensors="pt")
    with torch.no_grad():
        out = caption_model.generate(**inputs)
    return caption_processor.decode(out[0], skip_special_tokens=True)


# Qdrant global collection
# client = QdrantClient(":memory:")
os.makedirs("qdrant_storage", exist_ok=True)
client = QdrantClient(path="qdrant_storage")
COLLECTION = "video_frames_global"


def create_collection(vector_size=512):
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config={"size": vector_size, "distance": "Cosine"},
    )


def insert_embeddings(embeddings, frame_paths, video_name):
    # Ensure collection exists
    if not client.collection_exists(COLLECTION):
        create_collection()

    import uuid
    points = []
    for i, emb in enumerate(embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={"video": video_name, "frame": frame_paths[i]}
            )
        )
    client.upsert(collection_name=COLLECTION, points=points)


def search_frames(query, top_k=3, selected_video=None):
    inputs = clip_processor(text=query, return_tensors="pt")
    with torch.no_grad():
        q_emb = clip_model.get_text_features(**inputs)
    q_emb = q_emb.squeeze().cpu().numpy().tolist()

    # Apply video filter if one selected
    if selected_video and selected_video != "All":
        search_filter = {"must": [{"key": "video", "match": {"value": selected_video}}]}
        results = client.search(
            collection_name=COLLECTION,
            query_vector=q_emb,
            limit=top_k,
            query_filter=search_filter,
        )
    else:
        results = client.search(
            collection_name=COLLECTION, query_vector=q_emb, limit=top_k
        )

    return [(res.payload["frame"], res.payload["video"]) for res in results]
