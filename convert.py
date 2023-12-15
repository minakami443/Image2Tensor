import numpy as np
import cv2
import torch
from typing import List
from PIL import Image
from torchvision import transforms


class Image2Tensor():
    def __init__(self, 
            trans2tensor:List=[transforms.ToTensor()],
            trans2image:List=[transforms.ToPILImage()]) -> None:
        self.trans2tensor = transforms.Compose(trans2tensor)
        self.trans2image = transforms.Compose(trans2image)

    def toTensor(self, image:Image.Image) -> torch.FloatTensor:
        return self.trans2tensor(image)

    def toImage(self, tensor:torch.FloatTensor) -> Image.Image:
        image = tensor.cpu().clone()
        return self.trans2image(image)
    
class CV2Tensor(Image2Tensor):
    def __init__(self, 
            trans2tensor:List=[transforms.ToTensor()]) -> None:
        self.trans2tensor = transforms.Compose(trans2tensor)

    def toTensor(self, image:np.ndarray) -> torch.FloatTensor:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image.transpose((2, 0, 1)))
        return tensor.float().div(255)

    def toImage(self, tensor:torch.FloatTensor) -> np.ndarray:
        tensor = tensor.mul(255).byte().cpu()
        image = tensor.numpy().transpose((1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    

if __name__ == "__main__":
    p2t = Image2Tensor()
    c2t = CV2Tensor()
    image_path = "sample/000000000785.jpg"
    
    image = Image.open(image_path).convert('RGB')
    tensor = p2t.toTensor(image).unsqueeze(0)
    batch_tensor = torch.stack([tensor,tensor], dim =1).squeeze(0)
    for i, _ in enumerate(batch_tensor):
        image = p2t.toImage(_)
        image.save(f"output/PIL_{str(i)}.jpg")

    image = cv2.imread(image_path)
    tensor =c2t.toTensor(image).unsqueeze(0)
    batch_tensor = torch.stack([tensor,tensor], dim =1).squeeze(0)
    for i, _ in enumerate(batch_tensor):
        image = c2t.toImage(_)
        cv2.imwrite(f"output/CV2_{str(i)}.jpg", image)