from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from fpdf import FPDF
import os

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

def extract_qa_and_create_pdf(data_folder, question, output_pdf_path):
    # Load data from the specified directory
    documents = SimpleDirectoryReader(data_folder).load_data()

    # Create an index from the documents
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    # Create a query engine from the index
    query_engine = index.as_query_engine()

    # Query the engine with the specified question
    response = query_engine.query(question)
    
    # Create a new PDF file
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Write the question and response to the PDF
    pdf.multi_cell(0, 10, f"Question: {question}")
    pdf.multi_cell(0, 10, f"Answer: {response}")
    
    # Output the PDF to the specified path
    pdf.output(output_pdf_path)

# Example usage:
data_folder = "data"  # Folder containing the PDFs
question = "What is Qualnet simulator?"
output_pdf_path = "output_qa.pdf"
extract_qa_and_create_pdf(data_folder, question, output_pdf_path)
