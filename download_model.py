from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import cv2
from RealESRGAN import RealESRGAN
import matplotlib.pyplot as plt
from blurgenerator import lens_blur_with_depth_map, gaussian_blur_with_depth_map

#DEVIDE AGNOSTIC CODE
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")

print('device:', device)

path = "/Users/hasibulhasan/Downloads/PXL_20240601_172045936~2.jpg"

image = cv2.imread(path)
image_data = np.copy(image)

#LOAD DPT MODEL
model_path="/Users/hasibulhasan/pytorch/depth_serve/model"
processor = DPTImageProcessor.from_pretrained(model_path)
model = DPTForDepthEstimation.from_pretrained(model_path)
model.to(device)
print("Successfully loaded DPT model")
#PRE-PRECESS IMAGE
inputs = processor(images=image, return_tensors="pt").to(device)
print("Successfully preprocessed image for DPT model")
#FEED INPUT INTO DPT MODEL
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth
print("Inference done for DPT Model")
#POSTPROCESS AND SHOW ORIGINAL DEPTH OUTPUT
output = predicted_depth.cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth_og_np = np.squeeze(formatted, axis=0)

#CREATE FAKE RGB(Stacked_Depth) FROM B&H DEPTH FOR RealESRGAN MODEL
depth_fake_rgb = np.stack((depth_og_np,)*3, axis=-1)
print("Successfully preprocessed output for RealESRGAN Model")

#LOAD RealESRGAN MODEL
model_scale = "4" #@param ["2", "4", "8"] {allow-input: false}
model = RealESRGAN(device, scale=int(model_scale))
model.load_weights(f'weights/RealESRGAN_x{model_scale}.pth')
print("Successfully loaded RealESRGAN model")
#PREDICT RealESRGAN
hr_image = model.predict(depth_fake_rgb)
print("Inference done for RealESRGAN Model")

def resize_to_4k(img):
    height = img.shape[0]
    width = img.shape[1]
    my_height = 0
    my_width = 0
    if height > width:
        my_height = 4000
        my_width = int(my_height/(height/width))
    else:
        my_width = 4000
        my_height = int(my_width/(width/height))
    return cv2.resize(img, (my_width, my_height), interpolation=cv2.INTER_CUBIC)

image_cv_4k = resize_to_4k(image)
print("Successfully resized Original image to 4k")

# Convert the PIL image to a NumPy array
hr_image_np = np.array(hr_image)

# Change color space from RGB to Gray
hr_image_opencv = cv2.cvtColor(hr_image_np, cv2.COLOR_RGB2GRAY)

depth_4k = cv2.resize(hr_image_opencv, (image_cv_4k.shape[1], image_cv_4k.shape[0]), interpolation=cv2.INTER_CUBIC) 
print("Successfully resized RealESRGAN output image to 4k")


gray_image = cv2.cvtColor(image_cv_4k, cv2.COLOR_BGR2GRAY)


guided_depth = cv2.ximgproc.guidedFilter(guide=gray_image, src=depth_4k, radius=8, eps=0.4)


def laplacian_of_gaussian_edge_detection(image, ksize=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    edges = cv2.Laplacian(blur, cv2.CV_16S, ksize=ksize)
    abs_edges = cv2.convertScaleAbs(edges)
    return abs_edges

def apply_guided_filter(image, depth_map, radius=8, eps=0.4, iterations=3):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    guided_depth = depth_map
    for i in range(iterations):
        guided_depth = cv2.ximgproc.guidedFilter(guide=gray_image, src=guided_depth, radius=radius, eps=eps)
    return guided_depth

edges = laplacian_of_gaussian_edge_detection(image_cv_4k, ksize=5)
_, binary_edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
edge_mask = binary_edges > 0
refined_depth = np.where(edge_mask, depth_4k, depth_4k)

refined_depth = apply_guided_filter(image_cv_4k, refined_depth, radius=8, eps=0.1, iterations=8)

print("Successfully refined the edges")

cv2.imwrite('image_cv_4k.png', image_cv_4k)
cv2.imwrite('depth_4k.png', depth_4k)
cv2.imwrite('refined_depth.png', refined_depth)





