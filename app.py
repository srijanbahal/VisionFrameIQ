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
from segment import generate_highlighted_region


st.set_page_config(page_title="VideoFrameIQ", page_icon="🎥", layout="wide")

st.markdown(
    "<h2 style='text-align:center;'>🎥 VideoFrameIQ</h2>", unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Multi-Video Semantic Retrieval via CLIP + BLIP + Qdrant</p>",
    unsafe_allow_html=True,
)
st.divider()

# Sidebar
st.sidebar.header("⚙️ Settings")
fps = st.sidebar.slider("Frames per second", 1, 5, 2)
top_k = st.sidebar.slider("Top frames to display", 1, 10, 4)
st.sidebar.markdown("### 🧠 Example Queries")
st.sidebar.write("- person walking\n- car moving\n- someone talking\n- dog running")

if "videos" not in st.session_state:
    st.session_state.videos = []
if "initialized" not in st.session_state:
    create_collection()
    st.session_state.initialized = True

# ============ UPLOAD & PREVIEW SECTION ============
st.subheader("📂 Upload Videos")
uploaded_files = st.file_uploader(
    "Select one or more short video clips (MP4)",
    type=["mp4"],
    accept_multiple_files=True,
)

upload_dir = "sample_vision/uploaded"
os.makedirs(upload_dir, exist_ok=True)

if uploaded_files:
    # Save uploaded videos locally
    saved_videos = []
    for video_file in uploaded_files:
        video_path = os.path.join(upload_dir, video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        video_name = os.path.splitext(video_file.name)[0]
        saved_videos.append((video_name, video_path))

    # Compact preview grid
    st.markdown("### 🎞️ Video Previews")
    cols = st.columns(3)
    for i, (vname, vpath) in enumerate(saved_videos):
        col = cols[i % 3]
        col.markdown(f"**{vname}**")
        col.video(vpath, format="video/mp4", start_time=0)

    # Process all videos together
    if st.button("⚙️ Process All Videos"):
        for vname, vpath in saved_videos:
            with st.spinner(f"Processing {vname} ..."):
                frames, video_name = extract_frames(vpath, fps=fps)
                embeddings = [get_embedding(f) for f in frames]
                embeddings = smooth_embeddings(embeddings)
                insert_embeddings(embeddings, frames, video_name)
                if video_name not in st.session_state.videos:
                    st.session_state.videos.append(video_name)
        st.success("✅ All videos processed and indexed successfully!")

# ============ RETRIEVAL SECTION ============
st.divider()
st.subheader("🔍 Search Video Frames")

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
                for i, (frame_path, vid_name) in enumerate(row):
                    caption = generate_caption(frame_path)
                    highlighted_img = generate_highlighted_region(frame_path, query)
                    cols[i].image(highlighted_img, caption=f"{vid_name} — {caption}", use_column_width=True)
