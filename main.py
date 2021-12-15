from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import uvicorn
from result import result_router
from predict import load_model

"""
app : FastAPI 메인 앱
result_router : /result 관련 router : Image inference에 관련한 API
"""
app = FastAPI()
app.include_router(result_router)
templates = Jinja2Templates(directory='./templates/')

@app.get("/")
def get_files_form(request: Request):
    """
    Jinja2Templates 사용 -> Post만 구현
    """
    return templates.TemplateResponse('file_upload.html',context={'request': request})

@app.on_event("startup")
async def startup_event():
    """
    프로그램 시작시 실행, 모델 로드
    """
    print("application Start!!!")
    load_model()

@app.on_event("shutdown")
def startdown_event():
    """
    프로그램 종료 시 실행(로그 저장하거나, 웨이트 저장, 뭐 그런용도)
    """
    print("Application Stop.")

if __name__ == '__main__':
    """
    main : module(파일이름)
    app : app 이름
    """
    uvicorn.run('main:app', port=3000, host='0.0.0.0', reload=True)