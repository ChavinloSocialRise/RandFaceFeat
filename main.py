import torch
import numpy as np
from diffusers import FluxInpaintPipeline
from PIL import Image, ImageDraw
from ultralytics import YOLO

image_path = "test.jpg"

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


class FaceRandomizerPipeline:
    def __init__(self):
        print("Loading YOLO model...")
        self.face_detector = YOLO("yolov8m.pt")
        print("Loading Flux model...")
        self.image_gen = FluxInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.float16
        )
        self.image_gen.to("cuda")
        self.image_gen.enable_model_cpu_offload()
        print("Flux model loaded")

    def stage1(self, image: Image.Image):
        img_array = np.array(image)

        results = self.face_detector(img_array, agnostic_nms=True, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        formatted_boxes = [(int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(class_id)) for box, class_id in zip(boxes, class_ids)]
        
        mask = Image.new('1', image.size, 0)
        draw = ImageDraw.Draw(mask)

        for box in formatted_boxes:
            draw.rectangle(box[:4], fill=1)

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

        left_point = (nose[0], left_eyebrow[3])
        right_point = (nose[2], right_eyebrow[3])

        draw.polygon([left_point, (nose[0], nose[3]), (left_eyebrow[0], left_eyebrow[3])], fill=1)
        draw.polygon([right_point, (nose[2], nose[3]), (right_eyebrow[2], right_eyebrow[3])], fill=1)

        return mask
    
    def stage2(self, image: Image.Image, mask: Image.Image):
        output = self.image_gen(
            prompt="A selfie of a woman, verification photo",
            image=image,
            mask_image=mask,
            strength=0.8
        )

        return output

    def __call__(self, image_path):
        image = Image.open(image_path)
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