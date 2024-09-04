from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from funcs import extract_qa, append_qa_to_pdf

class Question(BaseModel):
    question: str

app = FastAPI()
data_folder = "data"
output_pdf_path = "output_qa.pdf"

@app.get("/")
def read_root():
    return {"message": "Welcome to the Question-Answer PDF Generator API"}

@app.post("/ask")
async def ask_question(item: Question):
    question, answer = extract_qa(data_folder, item.question)
    result = append_qa_to_pdf(question, answer, output_pdf_path)
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
