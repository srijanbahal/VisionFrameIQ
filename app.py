import streamlit as st
from PIL import Image
import os
from utils import (
    extract_frames, get_embedding, smooth_embeddings,
    create_collection, insert_embeddings, search_frames, generate_caption
)

st.set_page_config(page_title="VideoFrameIQ", page_icon="🎥", layout="centered")

st.markdown("<h2 style='text-align:center;'>🎥 VideoFrameIQ</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Multimodal Video Understanding via CLIP + BLIP + Qdrant</p>", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("⚙️ Settings")
fps = st.sidebar.slider("Frames per second to extract", 1, 5, 2)
top_k = st.sidebar.slider("Top results to display", 1, 5, 3)
st.sidebar.markdown("### 🧠 Example Queries")
st.sidebar.write("- person walking")
st.sidebar.write("- car moving")
st.sidebar.write("- someone talking")
st.sidebar.write("- animal running")

with st.expander("ℹ️ About this project"):
    st.markdown("""
    **VideoFrameIQ** uses CLIP embeddings and Qdrant vector search  
    to retrieve the most relevant video frames for your text query.  
    BLIP captions are used to describe retrieved frames.
    """)

video_file = st.file_uploader("Upload a short video (MP4)", type=["mp4"])
query = st.text_input("Enter a text query (e.g. 'person walking', 'car moving')")

# if video_file:
#     video_path = f"uploaded_{video_file.name}"
#     with open(video_path, "wb") as f:
#         f.write(video_file.getbuffer())
#     st.video(video_path)

#     if st.button("Process Video"):
#         with st.spinner("Extracting frames and computing embeddings..."):
#             frames = extract_frames(video_path, fps=fps)
#             embeddings = [get_embedding(f) for f in frames]
#             embeddings = smooth_embeddings(embeddings)
#             create_collection()
#             insert_embeddings(embeddings, frames)
#         st.success(f"✅ Processed {len(frames)} frames successfully!")

if video_file:
    # Ensure upload directory exists
    upload_dir = "sample_vision/uploaded"
    os.makedirs(upload_dir, exist_ok=True)

    # Save uploaded video inside the upload directory
    video_path = os.path.join(upload_dir, video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())

    # Display video preview
    st.video(video_path)

    # Process the video into frames + embeddings
    if st.button("Process Video"):
        with st.spinner("Extracting frames and computing embeddings..."):
            frames = extract_frames(video_path, fps=fps)  # stores in frames/<video_name>/
            embeddings = [get_embedding(f) for f in frames]
            embeddings = smooth_embeddings(embeddings)
            create_collection()
            insert_embeddings(embeddings, frames)
        st.success(f"✅ Processed {len(frames)} frames successfully!")


if query and st.button("Search"):
    with st.spinner("Searching best-matching frames..."):
        results = search_frames(query, top_k=top_k)
        if not results:
            st.warning("No matching frames found. Try a different query.")
        else:
            cols = st.columns(len(results))
            for i, r in enumerate(results):
                caption = generate_caption(r)
                cols[i].image(Image.open(r), caption=caption, use_column_width=True)
