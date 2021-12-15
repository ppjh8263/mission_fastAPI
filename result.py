import torch
from predict import inference
from utils import transform_image
from fastapi import UploadFile, File, APIRouter
from models import *

result_router = APIRouter(prefix='/result')


@result_router.get("/", description="Inference Result Get", tags=['Result'])
async def get_all_result() -> List[InferenceResult]:
    """
    전체 Inference Result 확인
    """
    return inference_result


@result_router.get("/{result_id}", description="Inference Result Get", tags=['Result'])
async def get_result(result_id: UUID) -> Union[InferenceResult, dict]:
    """
    ID를 사용하여 특정 결과 확인
    """
    result = get_result_by_id(result_id=result_id)
    if not result:
        return {"message": "Can not find Inference Result, Check ID"}
    return result


def get_result_by_id(result_id: UUID) -> Optional[InferenceResult]:
    return next((result for result in inference_result if result.id == result_id), None)


@result_router.post("/", description="Trash Image Classification Inference", tags=['Result'])
async def make_result(files: List[UploadFile] = File(...)):
    """
    이미지 Inference, 여러 이미지 동시에 가능
    """
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for file in files:
        image_bytes = await file.read()
        image = transform_image(image_bytes).to(device)
        result = InferenceImage(name = file.filename, result=inference(image, device))
        results.append(result)

    new_result = InferenceResult(results=results)
    inference_result.append(new_result)
    return new_result