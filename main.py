import torch
import numpy as np
from diffusers import FluxInpaintPipeline, AutoPipelineForInpainting
from PIL import Image, ImageDraw, ImageFilter
from ultralytics import YOLO

image_path = "test.jpg"

def closest_divisible_by_8(x):
    return x- x%8

class FaceRandomizerPipeline:
    def __init__(self):
        print("Loading YOLO model...")
        self.face_detector = YOLO("yolov8l.pt")
        print("Loading Flux model...")
        self.image_gen = AutoPipelineForInpainting.from_pretrained("SG161222/RealVisXL_V4.0")
        self.image_gen.to("cuda")
        self.image_gen.enable_model_cpu_offload()
        print("Flux model loaded")

    def stage1(self, image: Image.Image):
        img_array = np.array(image)

        results = self.face_detector(img_array, agnostic_nms=True, verbose=True)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        formatted_boxes = [(int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(class_id)) for box, class_id in zip(boxes, class_ids)]
        
        # Create a grayscale mask instead of binary
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)

        for box in formatted_boxes:
            draw.rectangle(box[:4], fill=255)

        left_eyebrow = None
        right_eyebrow = None
        nose = None

        for box in formatted_boxes:
            if box[4] == 3:
                if left_eyebrow is None or box[0] < left_eyebrow[0]:
                    left_eyebrow = box

                if right_eyebrow is None or box[0] > right_eyebrow[0]:
                    right_eyebrow = box

            if box[4] == 1:
                nose = box

        print(left_eyebrow, right_eyebrow, nose)

        left_point = (nose[0], left_eyebrow[3])
        right_point = (nose[2], right_eyebrow[3])

        draw.polygon([left_point, (nose[0], nose[3]), (left_eyebrow[0], left_eyebrow[3])], fill=255)
        draw.polygon([right_point, (nose[2], nose[3]), (right_eyebrow[2], right_eyebrow[3])], fill=255)

        # Apply Gaussian blur to soften the mask
        blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=20))
        
        # Convert to binary mask
        binary_mask = blurred_mask.point(lambda x: 0 if x < 128 else 255, '1')

        return binary_mask
    
    def stage2(self, image: Image.Image, mask: Image.Image):
        output = self.image_gen(
            prompt="instagram photo, portrair photo of a 20 y.o girl, perfect face, natural skin, looking to the left",
            negative_prompt="octane render, render, drawing, anime, bad photo, bad photography, worst quality, low quality, blurry, bad teeth, deformed teeth, deformed lips, bad anatomy, bad proportions, deformed iris, deformed pupils, deformed eyes, bad eyes, deformed face, ugly face, bad face, deformed hands, bad hands, fused fingers, morbid, mutilated, mutation, disfigured",
            image=image,
            mask_image=mask,
            strength=0.4,
            height=image.height,
            width=image.width,
            num_inference_steps=50
        )

        return output.images[0]

    def __call__(self, image_path):
        image = Image.open(image_path)
        image = image.resize((closest_divisible_by_8(image.width), closest_divisible_by_8(image.height)))
        print("Stage 1...")
        mask = self.stage1(image.copy())
        print("Stage 2...")
        output = self.stage2(image.copy(), mask)
        print("Done")

        return output
    
def main():
    pipeline = FaceRandomizerPipeline()
    x = pipeline(image_path)
    x.save(f"{image_path}_new.png")

if __name__ == "__main__":
    main()
