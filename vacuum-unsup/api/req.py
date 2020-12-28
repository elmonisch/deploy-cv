import requests

url = 'http://127.0.0.1:8000/predictions/vacuum'

# headers = {
#     'accept': 'application/json',
#     'Content-Type': 'multipart/form-data',
# }

# files = {'file': ('good1.png', open('good1.png', 'rb'), "image/png")}

# response = requests.post(url, files=files)

# print(response.text)

#### base 64

import base64
from pydantic import BaseModel

class ImageBase64(BaseModel):
    image: str

# img_b64 = base64.b64encode('good1.png').decode('utf-8')

with open('images/good8.png', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')

    img = ImageBase64(image=img_b64)

response = requests.post(url, data=img.json())
print(response.text)