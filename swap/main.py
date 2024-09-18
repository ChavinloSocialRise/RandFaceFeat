import torch
import numpy as np
import sys
import os
import cv2
from diffusers import AutoPipelineForInpainting, DPMSolverSDEScheduler
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import face_detection

# PARAMETERS

#FACE_PROMPT = "instagram photo, portrait photo of a 20 y.o girl, perfect face, natural skin, looking to the camera"
PROMPT = "instagram photo, selfie photo of a 20 y.o girl, perfect face, natural skin, looking to the camera, mirror selfie, random pose"

NEGATIVE_PROMPT = "octane render, render, drawing, anime, bad photo, bad photography, worst quality, low quality, blurry, bad teeth, deformed teeth, deformed lips, bad anatomy, bad proportions, deformed iris, deformed pupils, deformed eyes, bad eyes, deformed face, ugly face, bad face, deformed hands, bad hands, fused fingers, morbid, mutilated, mutation, disfigured"

####

def correction(image: Image.Image) -> Image.Image:
    width, height = image.size
    new_width = (width // 8) * 8
    new_height = (height // 8) * 8
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS).convert("RGB")
    return resized_image

class FaceRandomizerPipeline:
    def __init__(self):
        self.image_gen = AutoPipelineForInpainting.from_pretrained(
            "SG161222/RealVisXL_V5.0",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        self.image_gen.scheduler = DPMSolverSDEScheduler.from_config(self.image_gen.scheduler.config)
        self.image_gen.safety_checker = None
        self.image_gen.to("cuda")
        self.image_gen.enable_xformers_memory_efficient_attention()

        self.detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    
    def __call__(self, image: Image.Image) -> Image.Image:
        image = correction(image)
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        det = self.detector.detect(img_cv2)
        print(det)
        xmin, ymin, xmax, ymax, confidence = det[0]

        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([xmin, ymin, xmax, ymax], fill=255)

        output = self.image_gen(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=image,
            mask_image=mask,
            strength=0.4,
            height=image.height,
            width=image.width,
            num_inference_steps=50
        )

        return output.images[0]

def main():
    pipeline = FaceRandomizerPipeline()
    results = pipeline(Image.open("/home/chavinlo/RandFaceFeat/dsd.png").convert("RGB"))

if __name__ == "__main__":
    main()