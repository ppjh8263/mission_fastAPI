import torch
from src.model import Model

CLASSES = [
    "Metal",
    "Paper",
    "Paperpack",
    "Plastic",
    "Plasticbag",
    "Styrofoam",
]
"""
model global 변수로 지정하여 사용, 시작시 load할수 있도록 만들어 놓음 
-> 모델 로드하는데 오래 걸리기 때문에 처음 로드해놓고 계속 사용
"""
model = None

def load_model(weight:str = './model/best.pt', model_config:str='./model/model.yml'):
    """
    main에서 시작시 자동 실행
    model optimization에서 사용했던 모델
    """
    global model
    if weight.endswith("ts"):
        model = torch.jit.load(weight)
    else:
        model = Model(model_config, verbose=True)
        model.load_state_dict(
            torch.load(weight, map_location=torch.device("cpu"))
        )


@torch.no_grad()
def inference(img, device) -> str:
    """
    model inference
    """
    global model
    model = model.to(device)
    model.eval()
    img = img.to(device)
    pred = model(img)
    pred = torch.argmax(pred)

    return CLASSES[int(pred.detach())]