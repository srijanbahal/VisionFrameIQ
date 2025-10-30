import streamlit as st
from PIL import Image
import os
from utils import (
    extract_frames,
    get_embedding,
    smooth_embeddings,
    create_collection,
    insert_embeddings,
    search_frames,
    generate_caption,
)
from segment import highlight_region_with_sam2
from qdrant_client.http import models
from utils import client, COLLECTION

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="VideoFrameIQ", page_icon="🎥", layout="wide")

st.markdown("<h2 style='text-align:center;'>🎥 VideoFrameIQ</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Multi-Video Semantic Retrieval with CLIP + BLIP + SAM2 + Qdrant</p>", unsafe_allow_html=True)
st.divider()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Settings")
fps = st.sidebar.slider("Frames per second", 1, 5, 2)
top_k = st.sidebar.slider("Top frames to display", 1, 10, 4)
st.sidebar.markdown("### 🧠 Example Queries")
example_queries = ["person walking", "car moving", "someone talking", "dog running"]
for q in example_queries:
    st.sidebar.write(f"- {q}")

# =====================================================
# STATE MANAGEMENT
# =====================================================
if "videos" not in st.session_state:
    st.session_state.videos = []
if "query_suggestions" not in st.session_state:
    st.session_state.query_suggestions = []
if "initialized" not in st.session_state:
    create_collection()
    st.session_state.initialized = True

# =====================================================
# QDRANT CHECK FUNCTION
# =====================================================
def video_already_in_qdrant(video_name: str) -> bool:
    """Checks if a video's frames are already indexed in Qdrant."""
    try:
        res = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="video", match=models.MatchValue(value=video_name))]
            ),
            limit=1
        )
        return len(res[0]) > 0
    except Exception as e:
        print(f"Error checking Qdrant: {e}")
        return False

# =====================================================
# SIMPLE QUERY SUGGESTION FUNCTION
# =====================================================
def extract_query_suggestions_from_captions(captions, top_n=5):
    """Takes a list of BLIP captions and extracts key phrases."""
    import re
    from collections import Counter
    words = []
    for c in captions:
        for w in re.findall(r"\b[a-zA-Z]{4,}\b", c.lower()):
            words.append(w)
    common = Counter(words).most_common(top_n)
    return [w for w, _ in common]

# =====================================================
# UPLOAD SECTION
# =====================================================
st.subheader("📂 Upload Videos")
uploaded_files = st.file_uploader(
    "Select one or more short video clips (MP4)",
    type=["mp4"],
    accept_multiple_files=True,
)

upload_dir = "sample_vision/uploaded"
os.makedirs(upload_dir, exist_ok=True)

if uploaded_files:
    saved_videos = []
    for video_file in uploaded_files:
        video_path = os.path.join(upload_dir, video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        video_name = os.path.splitext(video_file.name)[0]
        saved_videos.append((video_name, video_path))

    st.markdown("### 🎞️ Video Previews")
    cols = st.columns(3)
    for i, (vname, vpath) in enumerate(saved_videos):
        col = cols[i % 3]
        col.markdown(f"**{vname}**")
        col.video(vpath, format="video/mp4", start_time=0)

    # =====================================================
    # PROCESSING SECTION
    # =====================================================
    if st.button("⚙️ Process All Videos"):
        all_captions = []
        for vname, vpath in saved_videos:
            video_name = os.path.splitext(os.path.basename(vpath))[0]

            # Check if video already indexed
            if video_already_in_qdrant(video_name):
                st.info(f"⏩ Skipping **{video_name}** (already indexed)")
                continue

            with st.spinner(f"Processing {vname} ..."):
                frames, _ = extract_frames(vpath, fps=fps)
                embeddings = [get_embedding(f) for f in frames]
                embeddings = smooth_embeddings(embeddings)
                insert_embeddings(embeddings, frames, video_name)

                # Generate sample captions
                sample_frames = frames[:5]
                all_captions.extend([generate_caption(f) for f in sample_frames])
                st.session_state.videos.append(video_name)

        # Generate query suggestions
        if all_captions:
            st.session_state.query_suggestions = extract_query_suggestions_from_captions(all_captions)

        st.success("✅ All new videos processed and indexed successfully!")

# =====================================================
# QUERY SUGGESTIONS
# =====================================================
if st.session_state.query_suggestions:
    st.markdown("### 💡 Suggested Queries")
    st.write(", ".join([f"`{q}`" for q in st.session_state.query_suggestions]))
# ============ RETRIEVAL SECTION ============
st.divider()
st.subheader("🔍 Search Video Frames")

# --- sidebar SAM toggle
use_sam = st.sidebar.checkbox("Enable SAM Region Highlighting", value=False,
                              help="Uncheck to disable segmentation overlay (recommended on CPU)")

query = st.text_input("Enter a text query (e.g., 'person walking', 'car moving')")

if st.session_state.videos:
    selected_video = st.sidebar.selectbox(
        "Filter by video", ["All"] + st.session_state.videos
    )
else:
    selected_video = "All"

if query and st.button("🔎 Search"):
    with st.spinner("Searching best-matching frames..."):
        results = search_frames(query, top_k=top_k, selected_video=selected_video)
        if not results:
            st.warning("No matching frames found.")
        else:
            st.markdown("### 🖼️ Retrieved Frames")
            n_cols = 3
            rows = [results[i : i + n_cols] for i in range(0, len(results), n_cols)]
            for row in rows:
                cols = st.columns(len(row))
                for i, (frame_path, vid_name, score) in enumerate(row):
                    caption = generate_caption(frame_path)

                    # --- safe SAM bypass ---
                    if use_sam:
                        try:
                            from segment import highlight_region_with_sam2
                            highlighted_img = highlight_region_with_sam2(frame_path)
                        except Exception as e:
                            st.warning(f"SAM failed: {e}")
                            highlighted_img = Image.open(frame_path)
                    else:
                        highlighted_img = Image.open(frame_path)

                    cols[i].image(
                        highlighted_img,
                        caption=f"{vid_name} — {caption} (score: {score:.2f})",
                        use_column_width=True,
                    )
