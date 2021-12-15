from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional
from datetime import datetime

#DB 대신 리스트 사용, InferenceResult의 리스트
inference_result = []

class InferenceImage(BaseModel):
    """
    Inference한 이미지 하나를 위한 Class
    """
    id: UUID = Field(default_factory=uuid4)
    name: str
    result: str

class InferenceResult(BaseModel):
    """
    InferenceImage에 담은 이미지를 List로 저장하는 Class
    """
    id: UUID = Field(default_factory=uuid4)
    results: List[InferenceImage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)