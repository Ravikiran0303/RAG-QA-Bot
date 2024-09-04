import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from funcs import read_api_key, extract_qa, append_qa_to_pdf

# Initialize the FastAPI app
app = FastAPI()

# Ensure the templates directory exists and contains your HTML files
templates = Jinja2Templates(directory="templates")

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

data_folder = "data"  # Folder containing the PDFs
output_pdf_path = "output_qa.pdf"

# Dictionary to store question embeddings
question_embeddings = {}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    question, answer = extract_qa(data_folder, question)
    result = append_qa_to_pdf(question, answer, output_pdf_path, question_embeddings)
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "answer": answer})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
