import os
import streamlit as st
import requests
import pandas as pd
from uuid import uuid4
from dotenv import load_dotenv
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from duckduckgo_search import DDGS
from ingest import ingest_data_from_folder, query_vector_store
from bs4 import BeautifulSoup
from groq import Groq

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = "data"
VECTOR_STORE_DIR = "vector_store"
os.makedirs(DATA_DIR, exist_ok=True)

# Get API keys from .env
LLAMA3_API_KEY = os.getenv("LLAMA3_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Groq setup
groq_model = Groq(api_key=GROQ_API_KEY)

# LangChain LLM wrapper
class LangChainLLM(BaseLLM):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _call(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"prompt": prompt, "max_tokens": 150}
        response = requests.post(
            "https://api.langchain.com/v1/completion", headers=headers, json=data
        )
        response_data = response.json()
        return response_data.get("choices", [{}])[0].get("text", "No response")

llama_model = LangChainLLM(api_key=LANGCHAIN_API_KEY)

# ✅ Fetch Tables with Titles Function
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

# Streamlit UI
st.set_page_config(page_title="Chatbot for Table Insights")
st.title("💬 Chatbot for Table Insights")

# -------------------------------
# 🔍 URL Input & Fetching Tables
# -------------------------------
url = st.text_input("Enter a URL to extract tables from:")

if st.button("Fetch Tables"):
    if url:
        with st.spinner("Fetching tables and saving to disk..."):
            all_tables = fetch_tables_with_titles(url)

            if not all_tables:
                st.warning("No tables found.")
            else:
                for table_info in all_tables:
                    st.subheader(f"📌 {table_info['title']}")
                    st.dataframe(table_info["dataframe"], use_container_width=True)

                    st.download_button("Download CSV", open(table_info["csv_path"], "rb"), file_name=os.path.basename(table_info["csv_path"]), mime="text/csv")
                    st.download_button("Download Excel", open(table_info["xlsx_path"], "rb"), file_name=os.path.basename(table_info["xlsx_path"]), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                with st.spinner("Running ingestion pipeline..."):
                    ingest_data_from_folder(DATA_DIR, VECTOR_STORE_DIR)
                st.success("✅ Ingestion completed. Vector store is updated.")
    else:
        st.warning("Please enter a valid URL.")

# -------------------------------
# 💬 Chatbot Q&A Section
# -------------------------------
question = st.text_input("Ask a question related to the data:")

if question:
    with st.spinner("Processing your query..."):
        answer_from_db = query_vector_store(question, VECTOR_STORE_DIR)

        if answer_from_db:
            st.write(f"Answer from Vector Store: {answer_from_db}")
        else:
            st.write("No relevant data found in vector store. Searching the internet...")

            # DuckDuckGo fallback
            search_results = DDGS(question)
            if search_results:
                st.write("Search Results from DuckDuckGo:")
                for result in search_results:
                    st.write(f"- {result['title']} ({result['url']})")

                combined_prompt = f"User asked: {question}\n\nContext from DuckDuckGo:\n"
                for result in search_results:
                    combined_prompt += f"{result['title']} - {result['url']}\n"

                if answer_from_db:
                    combined_prompt += f"\nContext from vector store: {answer_from_db}"

                try:
                    langchain_response = llama_model._call(combined_prompt)
                    st.write(f"Chatbot Answer: {langchain_response}")
                except Exception as e:
                    st.error(f"Error generating response: {e}")
            else:
                st.write("No search results found.")

# -------------------------------
# 🔁 Backup Model Query (Groq)
# -------------------------------
def query_llama_or_grok(prompt):
    try:
        response = llama_model._call(prompt)
        if not response:
            response = groq_model.query(prompt)
        return response
    except Exception as e:
        return f"Error during querying: {e}"
