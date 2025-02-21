import yolov5
import cv2
import numpy as np
from tkinter import Tk, filedialog

def upload_image():
    Tk().withdraw()  
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    return file_path

model = yolov5.load('keremberke/yolov5m-construction-safety')


model.conf = 0.20
model.iou = 0.40 
model.agnostic = False  
model.multi_label = False 
model.max_det = 1000 


img = upload_image()


results = model(img, size=640)

results = model(img, augment=True)

predictions = results.pred[0]
boxes = predictions[:, :4] 
scores = predictions[:, 4]
categories = predictions[:, 5]

results.ims = [np.copy(results.ims[0])] 

# Show results
results.show()

# Save results into "results/" folder
# results.save(save_dir='results/')
