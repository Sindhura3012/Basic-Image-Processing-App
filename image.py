import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time

st.title("Basic Image Processing App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    # Read image
    img_pil = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    st.subheader("Original Image")
    st.image(img_pil)

    # Q3 - Black & White
    bw = img_pil.convert("L")
    st.subheader("Black & White Image")
    st.image(bw)

    # Q4 - Properties
    st.subheader("Image Properties")
    st.write("Width & Height:", img_pil.size)
    st.write("Total Pixels:", img_pil.size[0] * img_pil.size[1])

    # File size in MB
    uploaded_file.seek(0, os.SEEK_END)
    size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)
    st.write("File Size (MB):", round(size_mb, 3))

    # Timestamp
    st.write("Uploaded On:", time.ctime(time.time()))

    # Q5 - Rotations
    st.subheader("Rotated Images")
    st.image(img_pil.rotate(-90), caption="Rotate 90°")
    st.image(img_pil.rotate(180), caption="Rotate 180°")
    st.image(img_pil.rotate(-270), caption="Rotate 270°")

    # Q6 - Mirror Image
    mirror = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
    st.subheader("Mirror Image")
    st.image(mirror)

    # Q7 - Object Detection (edges)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    st.subheader("Object Detection (Edges)")
    st.image(edges)

    # Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    st.write("Objects Detected:", len(contours))

    # Q8 - Vertical Cuts 80–20 & 90
    h, w, c = img_cv.shape
    cut80_left = img_cv[:, :int(w * 0.80)]
    cut20_right = img_cv[:, int(w * 0.80):]
    cut90 = img_cv[:, :int(w * 0.90)]

    st.subheader("Vertical Cuts")
    st.image(cut80_left, caption="80% Left")
    st.image(cut20_right, caption="20% Right")
    st.image(cut90, caption="90% Cut")

    # Q9 - Horizontal Cuts 70–30
    top70 = img_cv[:int(h * 0.70), :]
    bottom30 = img_cv[int(h * 0.70):, :]

    st.subheader("Horizontal Cuts")
    st.image(top70, caption="Top 70%")
    st.image(bottom30, caption="Bottom 30%")

    # Q10 - 3x3 Grid
    st.subheader("3x3 Grid Images")
    rows, cols = 3, 3
    grid_h = h // rows
    grid_w = w // cols

    for i in range(rows):
        for j in range(cols):
            grid = img_cv[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            st.image(grid, caption=f"Grid {i},{j}")

    # Reduced image <5MB
    small_img = img_pil
    small_img.save("reduced.jpg", quality=40)
    st.subheader("Reduced Image (<5MB)")
    st.image("reduced.jpg")
