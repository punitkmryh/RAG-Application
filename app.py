import os
from uuid import uuid4
import requests
import pandas as pd
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.chains import RetrievalQA

# For Groq chatbot
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- TABLE SCRAPING ----------------
def fetch_tables_with_titles(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        tables = soup.find_all("table")
        all_data = []

        for idx, table in enumerate(tables):
            parent = table.find_parent()
            title_tag = parent.find_previous(lambda tag: tag.name in ["h1", "h2", "h3", "h4", "h5", "h6", "span"] and tag.get("class") and "title" in tag.get("class"))

            title = title_tag.get_text(strip=True) if title_tag else f"table_{idx+1}"
            df = pd.read_html(str(table))[0]

            filename_base = f"{title.replace(' ', '_').lower()}_{uuid4().hex[:6]}"
            csv_path = os.path.join(DATA_DIR, f"{filename_base}.csv")
            xlsx_path = os.path.join(DATA_DIR, f"{filename_base}.xlsx")

            df.to_csv(csv_path, index=False)
            df.to_excel(xlsx_path, index=False)

            all_data.append({
                "title": title,
                "dataframe": df,
                "csv_path": csv_path,
                "xlsx_path": xlsx_path
            })

        return all_data
    except Exception as e:
        st.error(f"Failed to fetch tables: {e}")
        return []

# ---------------- VECTOR INGESTION ----------------
# ---------------- VECTOR INGESTION ----------------
def ingest_table(df):
    # Convert all columns to strings to avoid validation errors
    df = df.applymap(str)  # Apply str conversion to all elements in the dataframe

    loader = DataFrameLoader(df, page_content_column=df.columns[0])
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Uses CPU by default now
    db = FAISS.from_documents(split_docs, embeddings)
    return db


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Table Scraper + Q&A + Groq Chatbot", layout="wide")
st.title("📊 Table Scraper + 🧠 LLM Q&A + 🤖 Groq Chatbot")

# URL Input
url = st.text_input("Enter a webpage URL containing HTML tables")

if url:
    table_data = fetch_tables_with_titles(url)
    if table_data:
        st.success(f"Found {len(table_data)} table(s)!")
        for entry in table_data:
            st.subheader(entry['title'])
            st.dataframe(entry['dataframe'])
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("Download CSV", data=open(entry['csv_path'], 'rb'), file_name=os.path.basename(entry['csv_path']))
            with col2:
                st.download_button("Download Excel", data=open(entry['xlsx_path'], 'rb'), file_name=os.path.basename(entry['xlsx_path']))

        selected_title = st.selectbox("Select a table to chat with:", [t["title"] for t in table_data])
        selected_df = next(t for t in table_data if t["title"] == selected_title)["dataframe"]

        qa_question = st.text_input(f"Ask a question about '{selected_title}'")

        if qa_question:
            with st.spinner("Ingesting and thinking..."):
                db = ingest_table(selected_df)
                qa_chain = RetrievalQA.from_chain_type(llm=ChatGroq(model="gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]),
                                                       retriever=db.as_retriever())
                response = qa_chain.run(qa_question)
                st.success("Answer:")
                st.write(response)

# ---------------- GROQ CHATBOT ----------------
st.divider()
st.subheader("💬 General Groq Chatbot")

input_text = st.text_input("What question do you have in mind?", key="general_q")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the question."),
    ("user", "Question: {question}")
])

if input_text:
    try:
        # Change the model to an available one
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",  # Use a different, available model
            api_key=os.environ["GROQ_API_KEY"]
        )
        chain = prompt | llm | StrOutputParser()
        with st.spinner("Thinking..."):
            response = chain.invoke({"question": input_text})
        st.success("Response:")
        st.write(response)
    except Exception as e:
        st.error(f"Error generating response: {e}")
        st.write("Please check your model name or API access.")
