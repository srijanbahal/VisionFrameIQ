# VisionFrameIQ: Multimodal Video Understanding & Retrieval

**"Bridging text, vision, and segmentation for intelligent video insights."**

## 🚀 Overview

VisionFrameIQ is a multimodal video intelligence system designed to:

- Extract video frames efficiently and embed them using CLIP for semantic search.
- Retrieve the most relevant frames for text queries (e.g., "person smiling while driving").
- Apply region highlighting using Meta's Segment Anything 2.1 (SAM 2.1) to localize query-relevant areas.
- Optionally refine masks with CLIPSeg for context-aware segmentation.
- Provide an interactive Streamlit-based interface for search, visualization, and progressive feedback.

## ✨ Key Features

| Capability | Description |
|------------|-------------|
| 🎥 **Video Frame Extraction** | Extracts frames from uploaded videos at a configurable FPS. |
| 🧩 **Multimodal Embeddings (CLIP)** | Generates frame-level embeddings for semantic retrieval. |
| 🔍 **Text-to-Video Search** | Finds frames most semantically similar to a natural language query. |
| 🧠 **Qdrant Vector Store** | Stores and retrieves embeddings efficiently with metadata. |
| 🪄 **Region Highlighting (SAM 2.1)** | Highlights the most relevant regions using Segment Anything 2.1. |
| 🧾 **CLIPSeg Refinement** | Optional fine-grained refinement of the SAM masks using CLIPSeg. |
| 🖥️ **Interactive UI (Streamlit)** | Upload, process, and search across videos visually. |

## 🏗️ Current Architecture

```
VisionFrameIQ/
│
├── app.py                  # Streamlit interface (upload, search, display)
├── utils.py                # Frame extraction, embeddings, Qdrant helper functions
├── segment.py              # CLIPSeg + SAM2.1-based region highlighting
├── requirements.txt        # Dependencies
├── qdrant_storage/         # Local persistent Qdrant store
├── frames/                 # Extracted frames per video
│   ├── test01/
│   └── test02/
└── sample_videos/          # Uploaded video storage
```

## ⚙️ Setup Instructions

### 1️⃣ Clone & Setup Environment

```bash
git clone https://github.com/<your-username>/VisionFrameIQ.git
cd VisionFrameIQ

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2️⃣ Install Dependencies

Ensure you have:

```bash
pip install torch torchvision --upgrade
pip install opencv-python pillow
pip install streamlit qdrant-client
pip install transformers supervision
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 3️⃣ Run the App

```bash
streamlit run app.py
```

You can now:

- Upload videos (.mp4, .mov, etc.)
- Process & embed frames
- Enter text queries to find relevant video moments
- View highlighted regions (via SAM 2.1) directly in the Streamlit UI

## 🧩 How SAM 2.1 and CLIPSeg Work Here

### CLIP Embedding (retrieval)

- Each frame is encoded using a CLIP vision encoder; queries are encoded using the text encoder.
- Similarities are computed in the embedding space to find matching frames.

### SAM 2.1 (localization)

- Once frames are retrieved, SAM 2.1 segments the most relevant regions for visualization.

### CLIPSeg (refinement)

- CLIPSeg optionally refines masks using the text query, providing semantic alignment for finer visual correspondence.

## 📸 Example Usage

```python
from segment import highlight_region_with_sam2

result = highlight_region_with_sam2("frames/test01/frame_0004.jpg")
result.show()
```

Or directly through the Streamlit app by uploading and querying.

## 🧠 Tech Stack

| Layer | Tools |
|-------|-------|
| **Retrieval Engine** | OpenAI CLIP / LLaVA embeddings, Qdrant |
| **Segmentation** | SAM 2.1 (facebook/sam2-hiera-base-plus), CLIPSeg |
| **Frontend** | Streamlit (currently), React (planned) |
| **Backend** | FastAPI / Python |
| **Storage** | Qdrant (vector DB), local persistent storage |
| **Deployment** | Streamlit Cloud / Render (planned) |

## 🧪 Sample Queries

- "Person waving hand near car"
- "Dog jumping on sofa"
- "Logo appearing on the screen"
- "Close-up of smiling face"

## 🧭 Performance & Notes

- Qdrant persistence ensures embeddings survive Streamlit reloads.
- SAM 2.1's BF16 inference provides faster GPU performance.
- Combining CLIPSeg + SAM improves localization accuracy.
- Works efficiently on a mid-tier GPU (e.g., RTX 3060 12 GB).

## 🧱 Future Roadmap & TODOs

### 🧩 Immediate TODOs

- [ ] Integrate SAM 2.1 region highlighting into top-K frame retrieval loop (automated pipeline).
- [ ] Implement CLIPSeg-guided mask refinement for better text alignment.
- [ ] Add query-refinement feedback loop: learn from user-marked relevant frames.
- [ ] Multi-video retrieval ranking: merge results from multiple videos.

### 🧠 Model & Retrieval Refinements

- [ ] Add hybrid embedding fusion (visual + textual weighting).
- [ ] Introduce temporal continuity for neighboring frames (context-aware retrieval).
- [ ] Implement subclip extraction — generate short GIFs / 5-sec clips from retrieved frames.
- [ ] Add region-based scoring visualization (heatmaps).

### ⚙️ Backend TODOs

- [ ] Build FastAPI routes:
  - `/upload_video`
  - `/process_video`
  - `/search`
  - `/feedback`
- [ ] Make modular service layer for embeddings, Qdrant, and SAM.
- [ ] Create a `config.py` to unify paths, model IDs, and environment variables.

### 💻 Frontend TODOs

- [ ] React-based frontend using Next.js or Vite.
- [ ] Integrate video preview, frame timeline slider, and region overlays.
- [ ] Connect React UI to FastAPI backend (replacing Streamlit).
- [ ] Support drag-and-drop video uploads.
- [ ] Add search history & feedback dashboard.

### ☁️ Deployment TODOs

- [ ] Containerize via Docker.
- [ ] Deploy backend (FastAPI + Qdrant) on Render or Railway.
- [ ] Host frontend on Vercel or Netlify.
- [ ] Optionally integrate with Hugging Face Spaces (Gradio) for public demo.

### 🔬 Stretch Goals

- [ ] Introduce video-LLM commentary (text summary of matched subclips).
- [ ] Add multilingual query support (translate + encode).
- [ ] Incorporate Reinforcement via Feedback (RLFH) for retrieval quality tuning.
- [ ] Connect to LlamaIndex or ColPali for structured multimodal reasoning.
- [ ] Build dataset versioning and evaluation pipeline for fine-tuning.

## 🏁 Final Vision

**"A unified multimodal system that understands, retrieves, and explains video content in human language — combining perception, retrieval, and reasoning."**
