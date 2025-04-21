import os
import requests
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from io import BytesIO
from uuid import uuid4

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.schema import Document

# ------------------ Setup ------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="🧠 TableSense AI", layout="wide")
st.title("📊 Table Scraper and Chatbot for Table Insight Using LLM QA")

# Load Groq API Key from env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ------------------ Table Fetcher ------------------
def fetch_tables_with_titles(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all("table")
        all_data = []
        all_documents = []

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
                "xlsx_path": xlsx_path,
                "filename_base": filename_base
            })

            content = df.to_csv(index=False)
            document = Document(page_content=content, metadata={"source": filename_base})
            all_documents.append(document)

        return all_data, all_documents

    except Exception as e:
        st.error(f"Failed to fetch tables: {e}")
        return [], []

# ------------------ Build Vector DB + QA Chain ------------------
def build_qa_chain(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    db = FAISS.from_documents(chunks, embeddings)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
    return qa, db

# ------------------ DuckDuckGo Fallback Search ------------------
def duckduckgo_search(query):
    url = f"https://duckduckgo.com/html/?q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    results = []
    for result in soup.find_all("a", class_="result__a"):
        results.append(result.text.strip())

    return results

# ------------------ Step 1: Input URL ------------------
url = st.text_input("🔗 Enter webpage URL to scrape tables:", value="https://www.iitsystem.ac.in/mhrdprojects")

# ------------------ Step 2: Ingest Tables ------------------
if st.button("🚀 Ingest and Preview Tables"):
    with st.spinner("Fetching and processing tables..."):
        try:
            tables, documents = fetch_tables_with_titles(url)
            qa_chain, vectorstore = build_qa_chain(documents)

            st.session_state["qa_chain"] = qa_chain
            st.session_state["vectorstore"] = vectorstore
            st.session_state["tables"] = tables
            st.session_state["documents"] = documents

            st.success(f"✅ {len(tables)} tables ingested and ready for Q&A!")

        except Exception as e:
            st.error(f"❌ Failed to process URL: {str(e)}")

# ------------------ Step 3: Show Tables and Q&A Input ------------------
if "tables" in st.session_state and "qa_chain" in st.session_state:
    st.markdown("## 📄 Table Previews & Downloads")
    for idx, table_data in enumerate(st.session_state["tables"]):
        title = table_data["title"]
        df = table_data["dataframe"]
        filename_base = table_data["filename_base"]

        with st.expander(f"📌 {title}"):
            st.dataframe(df)

            col1, col2 = st.columns(2)
            with col1:
                with open(table_data["csv_path"], "rb") as f:
                    st.download_button("⬇️ Download CSV", f, file_name=f"{filename_base}.csv")

            with col2:
                with open(table_data["xlsx_path"], "rb") as f:
                    st.download_button("⬇️ Download XLSX", f, file_name=f"{filename_base}.xlsx")

    # Q&A Section
    st.markdown("---")
    st.subheader("💬 Ask a Question About the Data")

    question = st.text_input("Type your question here:")
    if st.button("Ask Question"):
        if question.strip():
            with st.spinner("🧠 Thinking..."):
                try:
                    qa_chain = st.session_state["qa_chain"]
                    result = qa_chain.run(question)
                    st.markdown(f"**🧠 Answer:** {result}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please enter a question.")

    # Internet Search
    st.markdown("---")
    st.subheader("🌐 Internet Search")

    search_query = st.text_input("Type something to search on the internet:")
    if st.button("Search Internet"):
        if search_query.strip():
            with st.spinner("Searching the web..."):
                try:
                    search_results = duckduckgo_search(search_query)
                    if search_results:
                        st.markdown("**🌍 Search Results:**")
                        for idx, result in enumerate(search_results):
                            st.write(f"{idx + 1}. {result}")
                    else:
                        st.warning("No search results found.")
                except Exception as e:
                    st.error(f"❌ Error during search: {str(e)}")
        else:
            st.warning("Please enter a search query.")
