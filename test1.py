import requests
import base64
from PIL import Image
from io import BytesIO

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image = Image.open(image_file).convert('RGB')
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

URL = "https://api.runpod.ai/v2/olax2f9c8kul8k/runsync"
TOKEN = "XWV1ST04C0QLWNVAUSJWI6VJMR7YDJCKJSAR6TPA"
IMAGE_PATH = "sample.jpg"

def send_image_to_api(image_path):
    base64_image = encode_image_to_base64(image_path)
    payload = {
        "input": {
            "image": base64_image
        }
    }
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.post(URL, json=payload, headers=headers)
    return response.json()

def main():
    result = send_image_to_api(IMAGE_PATH)
    print(result)
    # save returned image
    decoded_image = decode_base64_to_image(result["output"])
    decoded_image.save("decoded_image.jpg")

if __name__ == "__main__":
    main()