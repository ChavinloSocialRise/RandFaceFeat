import torch
import numpy as np
import sys
import os
import cv2
from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverSDEScheduler
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from PIL import Image
from controlnet_aux import OpenposeDetector

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inswapper.swapper import *
from inswapper.restoration import *

# PARAMETERS

#FACE_PROMPT = "instagram photo, portrait photo of a 20 y.o girl, perfect face, natural skin, looking to the camera"
PROMPT = "instagram photo, selfie photo of a 20 y.o girl, perfect face, natural skin, looking to the camera, mirror selfie, random pose"
NEGATIVE_PROMPT = "octane render, render, drawing, anime, bad photo, bad photography, worst quality, low quality, blurry, bad teeth, deformed teeth, deformed lips, bad anatomy, bad proportions, deformed iris, deformed pupils, deformed eyes, bad eyes, deformed face, ugly face, bad face, deformed hands, bad hands, fused fingers, morbid, mutilated, mutation, disfigured"

INSWAPPER_PATH = "inswapper_128.onnx"

####

def correction(image: Image.Image) -> Image.Image:
    width, height = image.size
    new_width = (width // 8) * 8
    new_height = (height // 8) * 8
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS).convert("RGB")
    return resized_image

class EndpointOnePipeline:
    def __init__(self):
        ## Standard Image Generator
        self.image_gen = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "SG161222/RealVisXL_V5.0",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        self.image_gen.scheduler = DPMSolverSDEScheduler.from_config(self.image_gen.scheduler.config)
        self.image_gen.safety_checker = None
        self.image_gen.to("cuda")
        self.image_gen.enable_xformers_memory_efficient_attention()

        ## Pose Conditioned Image Generator
        self.controlnet_m = ControlNetModel.from_pretrained(
            "xinsir/controlnet-openpose-sdxl-1.0",
            torch_dtype=torch.float16
        )

        self.control_gen = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=self.controlnet_m, torch_dtype=torch.float16
        )
        self.control_gen.enable_model_cpu_offload()

        self.detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

        # Face restoration initialization
        check_ckpts()
        self.upsampler = set_realesrgan()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"]
        ).to(self.device)
        ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
        checkpoint = torch.load(ckpt_path)["params_ema"]
        self.codeformer_net.load_state_dict(checkpoint)
        self.codeformer_net.eval()

    def generatePose(self, image: Image.Image) -> Image.Image:
        openpose_image = self.detector(image)

        poses = self.control_gen(
            prompt = PROMPT,
            negative_prompt = NEGATIVE_PROMPT,
            height = image.height,
            width = image.width,
            num_inference_steps = 35,
            controlnet_conditioning_scale = 0.5,
            image = openpose_image
        )

        return poses[0]
    
    def refineImage(self, image: Image.Image) -> Image.Image:
        output = self.image_gen(
            prompt = PROMPT,
            negative_prompt = NEGATIVE_PROMPT,
            height = image.height,
            width = image.width,
            image = image,
            num_inference_steps = 35,
            strength = 0.4
        )

        return output[0]

    def __call__(self, image: Image.Image) -> Image.Image:
        image = correction(image) # pose
        regenerated_pose = self.generatePose(image),
        refined_pose = self.refineImage(regenerated_pose)
        return refined_pose

def main():
    image_in = Image.open("tinder.jpg")
    pipeline = EndpointOnePipeline()
    results = pipeline(image_in)

if __name__ == "__main__":
    main()