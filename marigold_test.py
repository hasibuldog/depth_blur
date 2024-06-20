from diffusers import DiffusionPipeline
import numpy as np
import cv2
from PIL import Image

pipeline = DiffusionPipeline.from_pretrained("prs-eth/marigold-v1-0")

image_path="test_images/IMG_20200311_171140.jpg"

image = cv2.imread(image_path)
image_data = np.copy(image)
image_data = Image.open(image_path)

denoising_steps = 4 # @param {type:"integer"}
ensemble_size = 5 # @param {type:"integer"}
processing_res = 768 # @param {type:"integer"}
match_input_res = True # @param ["False", "True"]

pipeline_output = pipeline(
            image_data,
            denoising_steps=denoising_steps,     # optional
            ensemble_size=ensemble_size,       # optional
            processing_res=processing_res,     # optional
            match_input_res=match_input_res,   # optional
            batch_size=0,           # optional
            color_map="Spectral",   # optional
            show_progress_bar=True, # optional
        )
depth_pred = pipeline_output.depth_np
print(pipeline_output.shape)