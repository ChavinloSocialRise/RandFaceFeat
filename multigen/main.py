import torch
import numpy as np
import sys
import os
import cv2
from diffusers import StableDiffusionXLPipeline, DPMSolverSDEScheduler
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inswapper.swapper import *
from inswapper.restoration import *

# PARAMETERS

FACE_PROMPT = "instagram photo, portrait photo of a 20 y.o girl, perfect face, natural skin, looking to the camera"
POSES_PROMPT = "instagram photo, selfie photo of a 20 y.o girl, perfect face, natural skin, looking to the camera, mirror selfie, random pose"

NEGATIVE_PROMPT = "octane render, render, drawing, anime, bad photo, bad photography, worst quality, low quality, blurry, bad teeth, deformed teeth, deformed lips, bad anatomy, bad proportions, deformed iris, deformed pupils, deformed eyes, bad eyes, deformed face, ugly face, bad face, deformed hands, bad hands, fused fingers, morbid, mutilated, mutation, disfigured"

INSWAPPER_PATH = "inswapper_128.onnx"

####

class EndpointOnePipeline:
    def __init__(self):
        self.image_gen = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V5.0",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        self.image_gen.scheduler = DPMSolverSDEScheduler.from_config(self.image_gen.scheduler.config)
        self.image_gen.safety_checker = None
        self.image_gen.to("cuda")
        self.image_gen.enable_xformers_memory_efficient_attention()

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

    def generateFace(self) -> Image.Image:
        output = self.image_gen(
            prompt=FACE_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            height=1024,
            width=1024,
            num_inference_steps=35
        )

        output.images[0].save("face.png")

        return output.images[0]
    
    def generatePoses(self) -> list[Image.Image]:
        output = self.image_gen(
            prompt=POSES_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            height=1024,
            width=768,
            num_inference_steps=35,
            num_images_per_prompt=4
        )
        for i in range(len(output.images)):
            output.images[i].save(f"pose_{i}.png")
        return output.images
    
    def restore_face(self, image: Image.Image) -> Image.Image:
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        restored_array = face_restoration(
            img_array,
            background_enhance=True,
            face_upsample=True,
            upscale=2,
            codeformer_fidelity=0.5,
            upsampler=self.upsampler,
            codeformer_net=self.codeformer_net,
            device=self.device
        )
        return Image.fromarray(restored_array)

    def __call__(self) -> list[Image.Image]:
        base_face = self.generateFace()
        poses = self.generatePoses()

        result_list = []

        for pose in poses:
            result = process([base_face], pose, -1, -1, INSWAPPER_PATH)
            restored_result = self.restore_face(result)
            result_list.append(restored_result)

        return result_list

def main():
    pipeline = EndpointOnePipeline()
    results = pipeline()
    for i in range(len(results)):
        results[i].save(f"result_{i}.png")

if __name__ == "__main__":
    main()