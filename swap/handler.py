import base64
from io import BytesIO
from PIL import Image
from main import FaceRandomizerPipeline
import runpod

def handler(job):
    """ Handler function that will be used to process jobs. """
    global pipeline
    
    job_input = job['input']
    b64_image = job_input.get('image', '')
    image = Image.open(BytesIO(base64.b64decode(b64_image)))

    results = pipeline(image)
    buffered = BytesIO()
    results.save(buffered, format="JPEG", subsampling=0, quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

if __name__ == "__main__":
    global pipeline
    pipeline = FaceRandomizerPipeline()
    runpod.serverless.start({"handler": handler})
    