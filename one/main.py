import torch
import numpy as np
from diffusers import AutoPipelineForText2Image
from PIL import Image, ImageDraw, ImageFilter
from inswapper.swapper import *

# PARAMETERS

FACE_PROMPT = "instagram photo, portrait photo of a 20 y.o girl, perfect face, natural skin, looking to the camera"
POSES_PROMPT = "instagram photo, selfie photo of a 20 y.o girl, perfect face, natural skin, looking to the camera, mirror selfie, random pose"

NEGATIVE_PROMPT = "octane render, render, drawing, anime, bad photo, bad photography, worst quality, low quality, blurry, bad teeth, deformed teeth, deformed lips, bad anatomy, bad proportions, deformed iris, deformed pupils, deformed eyes, bad eyes, deformed face, ugly face, bad face, deformed hands, bad hands, fused fingers, morbid, mutilated, mutation, disfigured"

INSWAPPER_PATH = "inswapper_128.onnx"

####

class EndpointOnePipeline:
    def __init__(self):
        self.image_gen = AutoPipelineForText2Image.from_pretrained("SG161222/RealVisXL_V5.0")
        self.image_gen.to("cuda")
        self.image_gen.enable_xformers_memory_efficient_attention()
        self.image_gen.enable_model_cpu_offload()

    def generateFace(self) -> Image.Image:
        output = self.image_gen(
            prompt=FACE_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            height=1024,
            width=1024,
            num_inference_steps=4
        )

        return output.images[0]
    
    def generatePoses(self) -> list[Image.Image]:
        output = self.image_gen(
            prompt=POSES_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            height=1024,
            width=768,
            num_inference_steps=4,
            num_images_per_prompt=4
        )
        return output.images
    
    def __call__(self) -> list[Image.Image]:
        base_face = self.generateFace()
        poses = self.generatePoses()

        result_list = []

        for pose in poses:
            result = process([base_face], pose, -1, -1, INSWAPPER_PATH)
            result_list.append(result)

        return result_list