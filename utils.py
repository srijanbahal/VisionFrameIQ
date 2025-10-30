# utils.py
import os
import json
import uuid
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration,
)
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# ============================================================
# 🔧 Device setup (CPU safe)
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = ""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 🧠 Load Models
# ============================================================
print(f"Loading models on device: {DEVICE}")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# ============================================================
# 💾 Qdrant Client Initialization (Streamlit-safe)
# ============================================================
os.makedirs("qdrant_storage", exist_ok=True)

if "qdrant_client" not in globals():
    client = QdrantClient(path="qdrant_storage")
    print("✅ Qdrant client initialized.")
else:
    client = globals()["qdrant_client"]
    print("⚙️ Reusing existing Qdrant client.")

COLLECTION = "video_frames_global"


def create_collection(vector_size=512):
    """Create or reuse collection if not exists."""
    if not client.collection_exists(COLLECTION):
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config={"size": vector_size, "distance": "Cosine"},
        )


# ============================================================
# 🎥 Frame extraction + embeddings
# ============================================================
def extract_frames(video_path, output_folder="frames", fps=2):
    """Extract frames from video at specified FPS and store in per-video folder."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or fps
    interval = max(1, int(frame_rate // fps))

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
    frame_files = [os.path.join(video_folder, f) for f in sorted(os.listdir(video_folder))]
    return frame_files, video_name


def get_image_embedding_from_path(image_path):
    """Helper: get CLIP embedding for an image path."""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return emb.squeeze().cpu().numpy()


def get_embedding(image_path):
    """Alias for backward compatibility."""
    return get_image_embedding_from_path(image_path)


def get_text_embedding(text: str):
    """Get CLIP text embedding."""
    inputs = clip_processor(text=[text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
    return emb.squeeze().cpu().numpy()


def smooth_embeddings(embeddings, window=3):
    """Simple moving average smoothing for embeddings."""
    if len(embeddings) == 0:
        return embeddings
    smoothed = []
    for i in range(len(embeddings)):
        start = max(0, i - window)
        end = min(len(embeddings), i + window + 1)
        avg_emb = np.mean(embeddings[start:end], axis=0)
        smoothed.append(avg_emb)
    return smoothed


# ============================================================
# 🖋️ Caption generation
# ============================================================
def generate_caption(image_path):
    """Generate natural-language caption for a given frame."""
    image = Image.open(image_path).convert("RGB")
    inputs = caption_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = caption_model.generate(**inputs)
    return caption_processor.decode(out[0], skip_special_tokens=True)


# ============================================================
# 📈 Insert / Search
# ============================================================
def insert_embeddings(embeddings, frame_paths, video_name):
    """Insert frame embeddings into Qdrant collection."""
    create_collection()
    points = []
    for i, emb in enumerate(embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={"video": video_name, "frame": frame_paths[i]},
            )
        )
    client.upsert(collection_name=COLLECTION, points=points)


def search_frames(query, top_k=3, selected_video=None):
    """Search for frames matching text query."""
    q_emb = get_text_embedding(query).tolist()
    if selected_video and selected_video != "All":
        search_filter = {"must": [{"key": "video", "match": {"value": selected_video}}]}
        results = client.search(
            collection_name=COLLECTION,
            query_vector=q_emb,
            limit=top_k,
            query_filter=search_filter,
        )
    else:
        results = client.search(collection_name=COLLECTION, query_vector=q_emb, limit=top_k)

    return [(res.payload["frame"], res.payload["video"], res.score) for res in results]


def search_by_vector(vector, top_k=3, selected_video=None):
    """Search frames using precomputed embedding vector."""
    q_emb = vector.tolist() if isinstance(vector, np.ndarray) else vector
    if selected_video and selected_video != "All":
        search_filter = {"must": [{"key": "video", "match": {"value": selected_video}}]}
        results = client.search(
            collection_name=COLLECTION,
            query_vector=q_emb,
            limit=top_k,
            query_filter=search_filter,
        )
    else:
        results = client.search(collection_name=COLLECTION, query_vector=q_emb, limit=top_k)
    return [(res.payload["frame"], res.payload["video"], res.score) for res in results]


# ============================================================
# 👍👎 Feedback storage
# ============================================================
FEEDBACK_PATH = "feedback.json"


def load_feedback():
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"entries": []}


def save_feedback(feedback):
    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2)


def add_feedback(frame_path, label):  # label: "positive" or "negative"
    fb = load_feedback()
    emb = get_image_embedding_from_path(frame_path).tolist()
    fb["entries"].append({"frame": frame_path, "label": label, "embedding": emb})
    save_feedback(fb)


# ============================================================
# 🧮 Refinement logic
# ============================================================
def _mean_embedding(list_of_embs):
    if len(list_of_embs) == 0:
        return None
    return np.mean(np.stack(list_of_embs, axis=0), axis=0)


def normalize(vec):
    v = np.array(vec, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def refine_query_embedding(original_query, topk_frames_embeddings=None, feedback_ratio=(0.7, 0.2, 0.2)):
    """
    Blend formula:
      refined = α * q + β * pos_mean - γ * neg_mean
    feedback_ratio = (α, β, γ)
    """
    alpha, beta, gamma = feedback_ratio
    q_emb = get_text_embedding(original_query)

    fb = load_feedback()
    pos_embs = [np.array(e["embedding"]) for e in fb["entries"] if e["label"] == "positive"]
    neg_embs = [np.array(e["embedding"]) for e in fb["entries"] if e["label"] == "negative"]

    if len(pos_embs) == 0 and topk_frames_embeddings:
        pos_embs = topk_frames_embeddings[:min(len(topk_frames_embeddings), 3)]

    pos_mean = _mean_embedding(pos_embs) if len(pos_embs) > 0 else None
    neg_mean = _mean_embedding(neg_embs) if len(neg_embs) > 0 else None

    refined = np.array(q_emb, dtype=float)
    if pos_mean is not None:
        refined = refined * alpha + pos_mean * beta
    if neg_mean is not None:
        refined = refined - neg_mean * gamma

    return normalize(refined)


# ============================================================
# 📦 Helper: Retrieve embeddings for a list of frames
# ============================================================
def get_embeddings_for_frames(frame_paths):
    embs = []
    for p in frame_paths:
        embs.append(get_image_embedding_from_path(p))
    return embs
