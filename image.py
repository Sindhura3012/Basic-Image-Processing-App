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

    # Dropdown menu
    option = st.selectbox(
        "Choose an operation",
        [
            "Show Original Image",
            "Black & White",
            "Image Properties",
            "Rotate Image (90°, 180°, 270°)",
            "Mirror Image",
            "Object Detection (Edges)",
            "Vertical Cuts (80-20 & 90)",
            "Horizontal Cuts (70-30)",
            "3 x 3 Grid"
        ]
    )

    # Operation 1: Original
    if option == "Show Original Image":
        st.subheader("Original Image")
        st.image(img_pil)

    # Operation 2: Black & White
    if option == "Black & White":
        bw = img_pil.convert("L")
        st.subheader("Black & White Image")
        st.image(bw)

    # Operation 3: Properties
    if option == "Image Properties":
        st.subheader("Image Properties")
        st.write("Width & Height:", img_pil.size)
        st.write("Total Pixels:", img_pil.size[0] * img_pil.size[1])

        uploaded_file.seek(0, os.SEEK_END)
        size_mb = uploaded_file.tell() / (1024 * 1024)
        uploaded_file.seek(0)
        st.write("File Size (MB):", round(size_mb, 3))

        st.write("Uploaded On:", time.ctime(time.time()))

    # Operation 4: Rotations
    if option == "Rotate Image (90°, 180°, 270°)":
        st.subheader("Rotated Images")
        st.image(img_pil.rotate(-90), caption="Rotate 90°")
        st.image(img_pil.rotate(180), caption="Rotate 180°")
        st.image(img_pil.rotate(-270), caption="Rotate 270°")

    # Operation 5: Mirror
    if option == "Mirror Image":
        mirror = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        st.subheader("Mirror Image")
        st.image(mirror)

    # Operation 6: Object Detection
    if option == "Object Detection (Edges)":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        st.subheader("Edges / Object Detection")
        st.image(edges)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        st.write("Objects Detected:", len(contours))

    # Operation 7: Vertical Cuts
    if option == "Vertical Cuts (80-20 & 90)":
        h, w, c = img_cv.shape

        cut80_left = img_cv[:, :int(w * 0.80)]
        cut20_right = img_cv[:, int(w * 0.80):]
        cut90 = img_cv[:, :int(w * 0.90)]

        st.subheader("Vertical Cuts")
        st.image(cut80_left, caption="Left 80%")
        st.image(cut20_right, caption="Right 20%")
        st.image(cut90, caption="90% Cut")

    # Operation 8: Horizontal Cuts
    if option == "Horizontal Cuts (70-30)":
        h, w, c = img_cv.shape
        top70 = img_cv[:int(h * 0.70), :]
        bottom30 = img_cv[int(h * 0.70):, :]

        st.subheader("Horizontal Cuts")
        st.image(top70, caption="Top 70%")
        st.image(bottom30, caption="Bottom 30%")

    # Operation 9: Grid
    if option == "3 x 3 Grid":
        h, w, c = img_cv.shape
        rows, cols = 3, 3
        grid_h = h // rows
        grid_w = w // cols

        st.subheader("3 x 3 Grid")

        for i in range(rows):
            for j in range(cols):
                grid = img_cv[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                st.image(grid, caption=f"Grid {i}, {j}")
