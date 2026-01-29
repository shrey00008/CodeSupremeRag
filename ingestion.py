import langchain_huggingface
import langchain_google_genai
import faiss
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

PDF_FOLDER = "papers"
INDEX_PATH = "academic_faiss_index"

def run_ingestion():
    documents = []
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    
    print(f"Reading {len(pdf_files)} papers...")
    for filename in tqdm(pdf_files):
        loader = PyPDFLoader(os.path.join(PDF_FOLDER, filename))
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1050, 
        chunk_overlap=100
    )
    final_chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(final_chunks)}")

    print("Creating embeddings (this may take a few minutes)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(final_chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"✅ Success! Index saved to {INDEX_PATH}")

if __name__ == "__main__":
    run_ingestion()