import streamlit as st
from PIL import Image
import os
from utils import extract_frames, get_embedding, create_collection, insert_embeddings, search_frames

st.set_page_config(page_title="VideoFrameIQ", layout="centered")
st.title("🎥 VideoFrameIQ: Multimodal Video Understanding")

video_file = st.file_uploader("Upload a short video (MP4)", type=["mp4"])
query = st.text_input("Enter a text query (e.g. 'person walking', 'car moving')")

if video_file:
    video_path = f"uploaded_{video_file.name}"
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    st.video(video_path)

    if st.button("Process Video"):
        st.write("Extracting frames and computing embeddings...")
        frames = extract_frames(video_path)
        embeddings = [get_embedding(f) for f in frames]
        create_collection()
        insert_embeddings(embeddings, frames)
        st.success(f"Processed {len(frames)} frames successfully!")

if query and st.button("Search"):
    st.write("Searching best-matching frames...")
    results = search_frames(query)
    for r in results:
        st.image(Image.open(r), caption=os.path.basename(r))
