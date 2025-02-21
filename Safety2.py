import yolov5
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

def upload_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        return image
    return None

def draw_boxes_on_image(image, results, class_names):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for *box, conf, cls in results.pred[0]:
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(cls)]}: {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

model = yolov5.load('keremberke/yolov5m-construction-safety')

class_names = model.names


model.conf = 0.15  
model.iou = 0.30  
model.agnostic = False
model.multi_label = False
model.max_det = 1000

st.title("Construction Safety Detection")

img = upload_image()

if img is not None:
    results = model(img, size=960) 
    results = model(img, augment=True)

    img_with_boxes = draw_boxes_on_image(img, results, class_names)

    st.image(img_with_boxes, caption="Processed Image with Bounding Boxes", use_container_width=True)

    st.write(f"Detected {len(results.pred[0])} objects")
else:
    st.write("Please upload an image to perform detection.")
