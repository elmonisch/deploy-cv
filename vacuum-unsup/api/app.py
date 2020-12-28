from fastapi import FastAPI , File, UploadFile
from models import Resnet18FaceModel
from device import device
from torchvision import transforms
from pydantic import BaseModel
from PIL import Image
from torchvision.utils import save_image

import pickle
import torch
import io
import numpy as np
import time
import cv2
import base64

torch.manual_seed(0)
np.random.seed(0)

app = FastAPI()

###########################
### Load model
model = Resnet18FaceModel(num_classes=2)
state_dict = torch.load('vacuumc1.pth', map_location='cpu')

model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

### Preprocess image
tfms = transforms.Compose(
        [
            transforms.Resize((192,256)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )

with open("good.pkl", "rb") as f:
    embeding_good = pickle.load(f)
    embeding_good = torch.from_numpy(embeding_good)

class ImageBase64(BaseModel):
    image: str

@app.get("/")
async def ping():
    return "pong!"

# @app.post("/predictions/vacuum")
# async def inference(file: bytes = File(...)):
#     start_time = time.time()
#     img = Image.open(io.BytesIO(file))

#     if img.mode != "RGB":
#         img = img.convert("RGB")

#     img = tfms(img)
#     save_image(img, 'img1.png')
#     img = img.unsqueeze(0)
#     img = img.to(device)

#     with torch.no_grad():
#         _, embeding_new = model(img)

#     distance = torch.sum(torch.pow(embeding_good - embeding_new, 2))
    
#     if distance > 0.57:
#         status = "anomaly"
#     else:
#         status = "normal"

#     return {"status": f"{status}",
#             "device": f"{device}",
#             "distance": f"{distance}",
#             "inferenceTime": f"{round(time.time() - start_time, 4)}"}

@app.post("/predictions/vacuum")
async def inference(imageBase64: ImageBase64):
    # print(imageBase64.image)
    start_time = time.time()

    img = base64.b64decode(imageBase64.image)
    img = Image.open(io.BytesIO(img))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = tfms(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        _, embeding_new = model(img)

    distance = torch.sum(torch.pow(embeding_good - embeding_new, 2))
    distance = distance.detach().cpu().numpy()
    distance = round(float(distance), 4)

    if distance > 0.58:
        status = "anomaly"
    else:
        status = "normal"

    inferenceTime = round(time.time() - start_time, 4)

    return {"status": f"{status}",
            "device": f"{device}",
            "distance": f"{distance}",
            "inferenceTime": f"{inferenceTime}"}

# @app.post("/inference")
# async def inference(file: bytes = File(...)):
#     # print(embeding_good)
#     start_time = time.time()

#     # img = np.load(io.BytesIO(file), allow_pickle=True)
#     # print(img.shape)
#     # img = Image.frombytes('RGBA', (192, 256), file, 'raw')
#     img = Image.open(io.BytesIO(file)).convert('RGB')
#     print(f"{img.size}")
#     cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#     # print(cv_img)
#     print(f"cv2 shape : {cv_img.shape}")
#     cv2.imwrite('cv2.png', cv_img)
#     # img.resize((192, 256))
#     img.save("pil.png")
#     img = tfms(cv_img)
#     save_image(img, 'img1.png')
#     print(f"Shape : {img.shape}")
#     img = img.unsqueeze(0)
#     img = img.to(device)

#     with torch.no_grad():
#         _, embeding_new = model(img)

#     print(embeding_new)
#     distance = torch.sum(torch.pow(embeding_good - embeding_new, 2))
#     # print("distance: {}".format(distance))

#     if distance > 0.55:
#         status = "anomaly"
#     else:
#         status = "normal"

#     return {"status": f"{status}",
#             "device": f"{device}",
#             "distance": f"{distance}",
#             "inferenceTime": f"{time.time() - start_time}"}

# @app.post("/pic")
# async def pic(file: bytes = File(...)):

#     img = Image.open(io.BytesIO(file))

#     if img.mode != "RGB":
#         img = img.convert("RGB")

#     # img = img.resize((256, 192))
#     print(img.size)
#     img.save("pic.png")
#     # img_array = np.array(img)
#     # img_array = np.moveaxis(img_array, -1, 0)
#     # print(img_array.shape)

#     # img = torch.from_numpy(img_array)
#     # print(f"torch tensor : {img}")
#     # save_image(img, 'torchtensor.png')

#     tfms = transforms.Compose(
#         [
#             transforms.Resize((192,256)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
#         ]
#     )

#     img = tfms(img)
#     save_image(img, 'torchtensor.png')
#     # # img = 
#     print(img.shape)
#     img = img.unsqueeze(0)
#     img = img.to(device)

#     print(img.shape)

#     with torch.no_grad():
#         _, embeding_new = model(img)

#     # print(embeding_new)
#     distance = torch.sum(torch.pow(embeding_good - embeding_new, 2))
#     print("distance: {}".format(distance))


#     return {"file": f"test"}