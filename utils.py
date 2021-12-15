import torch
from PIL import Image
import torchvision.transforms as transforms
from src.augmentation.transforms import FILLCOLOR, SquarePad
import io

def transform_image(image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    image = Image.open(io.BytesIO(image))

    return transform(image).unsqueeze(0)
