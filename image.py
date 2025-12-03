import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os

st.title("Basic Image Processing App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    # Read image
    img_pil = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Sidebar options
    st.sidebar.title("Choose Operations")
    show_original = st.sidebar.checkbox("Show Original Image", True)
    show_bw = st.sidebar.checkbox("Convert to Black & White")
    show_properties = st.sidebar.checkbox("Show Image Properties")
    show_rotations = st.sidebar.checkbox("Rotate Image (90°, 180°, 270°)")
    show_mirror = st.sidebar.checkbox("Mirror Image")
    show_edges = st.sidebar.checkbox("Object Detection (Edges)")
    show_vertical_cuts = st.sidebar.checkbox("Vertical Cuts (80-20 & 90)")
    show_horizontal_cuts = st.sidebar.checkbox("Horizontal Cuts (70-30)")
    show_grid = st.sidebar.checkbox("3 x 3 Grid")

    # Show original
    if show_original:
        st.subheader("Original Image")
        st.image(img_pil)

    # Black & white
    if show_bw:
        bw = img_pil.convert("L")
        st.subheader("Black & White")
        st.image(bw)

    # Properties
    if show_properties:
        st.subheader("Image Properties")
        st.write("Width & Height:", img_pil.size)
        st.write("Total Pixels:", img_pil.size[0] * img_pil.size[1])

        uploaded_file.seek(0, os.SEEK_END)
        size_mb = uploaded_file.tell() / (1024 * 1024)
        uploaded_file.seek(0)
        st.write("File Size (MB):", round(size_mb, 3))

        st.write("Uploaded On:", time.ctime(time.time()))

    # Rotations
    if show_rotations:
        st.subheader("Rotated Images")
        st.image(img_pil.rotate(-90), caption="Rotate 90°")
        st.image(img_pil.rotate(180), caption="Rotate 180°")
        st.image(img_pil.rotate(-270), caption="Rotate 270°")

    # Mirror
    if show_mirror:
        mirror = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        st.subheader("Mirror Image")
        st.image(mirror)

    # Object Detection
    if show_edges:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        st.subheader("Edges / Object Detection")
        st.image(edges)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        st.write("Objects Detected:", len(contours))

    # Vertical Cuts
    if show_vertical_cuts:
        st.subheader("Vertical Cuts (80-20 & 90)")
        h, w, c = img_cv.shape

        cut80_left = img_cv[:, :int(w * 0.80)]
        cut20_right = img_cv[:, int(w * 0.80):]
        cut90 = img_cv[:, :int(w * 0.90)]

        st.image(cut80_left, caption="Left 80%")
        st.image(cut20_right, caption="Right 20%")
        st.image(cut90, caption="90% Cut")

    # Horizontal Cuts
    if show_horizontal_cuts:
        st.subheader("Horizontal Cuts (70-30)")
        h, w, c = img_cv.shape

        top70 = img_cv[:int(h * 0.70), :]
        bottom30 = img_cv[int(h * 0.70):, :]

        st.image(top70, caption="Top 70%")
        st.image(bottom30, caption="Bottom 30%")

    # 3x3 Grid
    if show_grid:
        st.subheader("3 x 3 Grid")
        h, w, c = img_cv.shape
        rows, cols = 3, 3
        grid_h = h // rows
        grid_w = w // cols

        for i in range(rows):
            for j in range(cols):
                grid = img_cv[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                st.image(grid, caption=f"Grid {i}, {j}")
