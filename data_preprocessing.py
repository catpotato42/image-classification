import os
import cv2
import numpy as np
import random
from glob import glob

# --- CONFIG ---
SOURCE_DIR = "./my_dataset"
EXPORT_DIR = "./augmented_data"
CLASSES = ["two_class", "twin_class", "timeout_class", "working_class", "five_class", "one_class", "crine_class", "thinking_class"]
# --------------

def apply_transforms(img):

    #random contrast and brightness
    contrast = random.uniform(0.5, 1.5)
    brightness = random.uniform(-50, 30) #goes to 100 but 30 is plenty bright
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness) # alpha * f(i,j) + beta

    #random compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(20, 50)]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    img = cv2.imdecode(encimg, 1)
    
    return img

for cls in CLASSES:
    os.makedirs(os.path.join(EXPORT_DIR, cls), exist_ok=True)
    images = glob(os.path.join(SOURCE_DIR, cls, "*.jpg"))
    
    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        #save original
        cv2.imwrite(os.path.join(EXPORT_DIR, cls, f"{i}.jpg"), img)
        #save augmented
        aug_img = apply_transforms(img)
        cv2.imwrite(os.path.join(EXPORT_DIR, cls, f"aug_{i}.jpg"), aug_img)