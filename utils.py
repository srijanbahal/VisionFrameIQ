import cv2, os, torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# --- Load CLIP once ---
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP model loaded.")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP processor loaded.")

# --- Extract frames from video ---
def extract_frames(video_path, output_folder="frames", fps=2):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval = int(frame_rate // fps) if frame_rate else 1
    i, count = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            count += 1
        i += 1
    cap.release()
    return [os.path.join(output_folder, f) for f in sorted(os.listdir(output_folder))]

# --- Get image embedding ---
def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return emb.squeeze().cpu().numpy().tolist()

# --- Setup Qdrant in-memory ---
client = QdrantClient(":memory:")
COLLECTION = "video_frames"

def create_collection(vector_size=512):
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config={"size": vector_size, "distance": "Cosine"}
    )

def insert_embeddings(embeddings, frame_paths):
    points = [
        PointStruct(id=i, vector=emb, payload={"frame": frame_paths[i]})
        for i, emb in enumerate(embeddings)
    ]
    client.upsert(collection_name=COLLECTION, points=points)

def search_frames(query, top_k=3):
    inputs = clip_processor(text=query, return_tensors="pt")
    with torch.no_grad():
        q_emb = clip_model.get_text_features(**inputs)
    q_emb = q_emb.squeeze().cpu().numpy().tolist()
    results = client.search(collection_name=COLLECTION, query_vector=q_emb, limit=top_k)
    return [res.payload["frame"] for res in results]



if __name__ == "__main__":
    print("Utils module for VideoFrameIQ loaded.")