import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from uuid import uuid4
from pathlib import Path
from io import StringIO  # Added for the future warning fix

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)


def extract_tables(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = pd.read_html(StringIO(response.text))  # Wrapped in StringIO to avoid the FutureWarning
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5'])

    header_texts = [h.get_text(strip=True) for h in headers]

    results = []
    for i, table in enumerate(tables):
        title = header_texts[i] if i < len(header_texts) else f"Table_{i+1}"
        safe_title = "_".join(title.strip().split()).replace("/", "_")

        csv_path = os.path.join(SAVE_DIR, f"{safe_title}.csv")
        xlsx_path = os.path.join(SAVE_DIR, f"{safe_title}.xlsx")

        # Explicitly specify the engine as 'openpyxl' for Excel files
        table.to_csv(csv_path, index=False)
        table.to_excel(xlsx_path, index=False, engine="openpyxl")

        doc_text = f"Title: {title}\n\n" + table.to_markdown()
        results.append(Document(page_content=doc_text, metadata={"source": title}))

    return results


def ingest(url):
    documents = extract_tables(url)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local("faiss_index")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ingest(sys.argv[1])
    else:
        print("❌ Please provide a URL. Example: python ingest_1.py <url>")
