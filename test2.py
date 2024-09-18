import requests
import base64
from PIL import Image
from io import BytesIO

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

URL = "https://api.runpod.ai/v2/9smkvsgnmjwn9v/runsync"
TOKEN = "XWV1ST04C0QLWNVAUSJWI6VJMR7YDJCKJSAR6TPA"
IMAGE_PATH = "sample.jpg"

def send_to_api():
    payload = {
        "input": {
            "x": "x"
        }
    }
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.post(URL, json=payload, headers=headers)
    return response.json()

def main():
    result = send_to_api()
    print(result)
    # save returned images
    for x in range(len(result["output"])):
        decoded_image = decode_base64_to_image(result["output"][x])
        decoded_image.save(f"decoded_image_{x}.jpg")

if __name__ == "__main__":
    main()