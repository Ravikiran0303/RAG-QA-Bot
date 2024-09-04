import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from fpdf import FPDF
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai

def read_api_key(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('OPENAI_API_KEY='):
                api_key = line.strip().split('=')[1].strip('"')
                return api_key
    return None

# Path to your key.txt file
key_file_path = r'C:\Ravikiran\Projects\RAG\key.txt'

# Read the API key
api_key = read_api_key(key_file_path)
os.environ["OPENAI_API_KEY"] = api_key

# Initialize OpenAI client
def get_openai_client(api_key):
    openai.api_key = api_key
    return openai

# Function to extract question embeddings
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = openai.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)

def extract_qa(data_folder, question):
    # Load data from the specified directory
    documents = SimpleDirectoryReader(data_folder).load_data()

    # Create an index from the documents
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    # Create a query engine from the index
    query_engine = index.as_query_engine()

    # Query the engine with the specified question
    response = query_engine.query(question)
    
    return question, response

def read_existing_pdf(pdf_path):
    existing_content = ""
    question_count = 0
    questions = {}
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                existing_content += text
                for line in text.split("\n"):
                    if line.startswith("Question "):
                        question_count += 1
                        question_text = line.split(":", 1)[1].strip()
                        questions[question_text.lower()] = question_count
    except FileNotFoundError:
        # If the file does not exist, return empty values
        pass
    return existing_content, question_count, questions

def append_qa_to_pdf(question, answer, pdf_path, question_embeddings):
    existing_content, question_count, questions = read_existing_pdf(pdf_path)

    # Compute the embedding for the new question
    question_embedding = get_embedding(question)

    # Check for semantic similarity
    for existing_question in questions.keys():
        existing_question_embedding = get_embedding(existing_question)
        similarity = cosine_similarity([question_embedding], [existing_question_embedding])[0][0]
        if similarity > 0.8:  # Threshold for considering questions as duplicates
            question_number = questions[existing_question.lower()]
            return f"The question is repeated, you can check question {question_number}"

    question_number = question_count + 1

    # Create a new PDF file
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Write the existing content to the PDF
    if existing_content:
        pdf.multi_cell(0, 10, existing_content)

    # Write the new question and response to the PDF with the question number
    pdf.multi_cell(0, 10, f"Question {question_number}: {question}")
    pdf.multi_cell(0, 10, f"Answer: {answer}")
    
    # Output the PDF to the specified path
    pdf.output(pdf_path)

    # Store the embedding for the new question
    question_embeddings[question.lower()] = question_embedding

    return f"Question {question_number} added successfully."