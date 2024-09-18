import cv2
import face_detection
from PIL import Image, ImageDraw
import numpy as np
print(face_detection.available_detectors)
detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
# BGR to RGB
# Open image and convert to BGR for OpenCV processing
im = Image.open("dsd.png").convert("RGB")
im_cv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

# Perform face detection
det = detector.detect(im_cv)
print(det)

# Create a drawing context on the image
draw = ImageDraw.Draw(im)

# Loop through each detection and draw a rectangle around it
xmin, ymin, xmax, ymax, confidence = det[0]

# Draw rectangle on the original image
draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

im.save("dsd2.jpg")