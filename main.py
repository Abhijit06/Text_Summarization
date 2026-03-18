# from fastapi import FastAPI, HTTPException, Request
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Optional
# import uvicorn

# from summarizer import summarize_text

# app = FastAPI(
#     title="Text Summarization API",
#     description="AI-powered text summarization using Hugging Face models",
#     version="1.0.0"
# )

# # ✅ FIXED: Added @ decorator
# @app.get("/")
# def root():
#     return {"message": "Hello World"}

# # ✅ FIXED: Middleware must be added before routes
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class SummarizeResponse(BaseModel):
#     summary: str
#     original_length: int
#     summary_length: int
#     model_used: str
#     compression_ratio: float

# class SummarizeRequest(BaseModel):
#     text: str
#     length: Optional[str] = "medium"
#     model: Optional[str] = "bart"

# # ✅ FIXED: Accepts both JSON and plain text, length & model as query params
# @app.post("/summarize", response_model=SummarizeResponse)
# async def summarize(request: Request, length: str = "medium", model: str = "bart"):
#     text = request.text
#     length: str = "medium",
#     model: str = "bart"

#     content_type = request.headers.get("content-type", "")

#     if "application/json" in content_type:
#         body = await request.json()
#         text = body.get("text", "")
#     else:
#         body = await request.body()
#         text = body.decode("utf-8")

#     if not text or len(text.strip()) < 50:
#         raise HTTPException(status_code=400, detail="Text must be at least 50 characters long.")

#     if len(text) > 100_000:
#         raise HTTPException(status_code=400, detail="Text exceeds maximum limit of 100,000 characters.")

#     try:
#         result = summarize_text(text=text, length=length, model_name=model)
#         return SummarizeResponse(
#             summary=result["summary"],
#             original_length=len(text),
#             summary_length=len(result["summary"]),
#             model_used=result["model_used"],
#             compression_ratio=round(len(result["summary"]) / len(text), 2)
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from summarizer import summarize_text  # your advanced function

app = FastAPI()


# Request body schema (better than query params)
class SummarizeRequest(BaseModel):
    text: str
    length: str = "medium"   # short | medium | long
    model_name: str = "bart" # bart | t5

import ftfy

def clean_text(text):
    return ftfy.fix_text(text)

@app.post("/summarize")
def summarize_api(request: SummarizeRequest):
    try:
        result = summarize_text(
            text=clean_text(request.text),
            length=request.length,
            model_name=request.model_name
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

    


# python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload