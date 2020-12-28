import logging
import torch
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler

class MyHandler(BaseHandler):
    """
    Custom handler for pytorch serve. This handler supports batch requests.
    For a deep description of all method check out the doc:
    https://pytorch.org/serve/custom_service.html
    """

    with open("good.pkl", "rb") as f:
        embeding_good = pickle.load(f)
        embeding_good = torch.from_numpy(embeding_good)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = transforms.Compose(
        [
            #transforms.ToPILImage(),
            transforms.Resize((192, 256)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )

    def preprocess_one_image(self, req):
        """
        Process one single image.
        """
        # get image from the request
        image = req.get("data")
        if image is None:
            image = req.get("body")       
         # create a stream from the encoded image
        image = Image.open(io.BytesIO(image))
        image = self.transform(image)
        # add batch dim
        image = image.unsqueeze(0)
        return image

    def preprocess(self, requests):
        """
        Process all the images from the requests and batch them in a Tensor.
        """
        images = [self.preprocess_one_image(req) for req in requests]
        images = torch.cat(images)    
        return images

    def inference(self, x):
        """
        Given the data from .preprocess, perform inference using the model.
        We return the predicted label for each image.
        """
        _, embeding_new = self.model.forward(x)
        # probs = F.softmax(outs, dim=1) 
        print("outs : ", outs)
        # preds = torch.argmax(outs, dim=1)
        # return preds

	distance = torch.sum(torch.pow(embeding_good - embeding_new, 2))
        # return outs
	
	return distance

    def postprocess(self, preds):
        """
        Given the data from .inference, postprocess the output.
        In our case, we get the human readable label from the mapping 
        file and return a json. Keep in mind that the reply must always
        be an array since we are returning a batch of responses.
        """
        res = []
        # pres has size [BATCH_SIZE, 1]
        # convert it to list
        preds = preds.cpu().tolist()
        for pred in preds:
            # label = self.mapping[str(pred)][1]
            # label = self.mapping[str(pred)]
            res.append({'prediction' : pred})
        return res
