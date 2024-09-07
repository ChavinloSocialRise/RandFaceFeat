import base64
from io import BytesIO
from main import EndpointOnePipeline
import runpod

def handler(job):
    """ Handler function that will be used to process jobs. """
    global pipeline
    results = pipeline()
    list_to_return = []
    for result in results:
        buffered = BytesIO()
        result.save(buffered, format="JPEG", subsampling=0, quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        list_to_return.append(img_str)
    return list_to_return

if __name__ == "__main__":
    global pipeline
    pipeline = EndpointOnePipeline()
    runpod.serverless.start({"handler": run})
    