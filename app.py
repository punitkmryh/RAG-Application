import os
import requests
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from io import BytesIO
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.schema import Document

# Set up paths
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="🧠 Table Q&A with LLM", layout="wide")
st.title("📊 Table Scraper + LLM QA")

# Setup Groq LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Fetch tables and convert to documents
def fetch_tables_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    tables = pd.read_html(response.text)

    documents = []
    for idx, table in enumerate(tables):
        title = f"Table_{idx + 1}"
        csv_path = os.path.join(DATA_DIR, f"{title}.csv")
        xlsx_path = os.path.join(DATA_DIR, f"{title}.xlsx")

        table.to_csv(csv_path, index=False)
        table.to_excel(xlsx_path, index=False)

        doc_text = f"Title: {title}\n\n" + table.to_markdown()
        documents.append(Document(page_content=doc_text, metadata={"source": title}))

    return tables, documents

# Build vectorstore and QA chain
def build_qa_chain(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    db = FAISS.from_documents(chunks, embeddings)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
    return qa, db

# Perform DuckDuckGo search
def duckduckgo_search(query):
    # Send query to DuckDuckGo
    url = f"https://duckduckgo.com/html/?q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract search result snippets
    results = []
    for result in soup.find_all("a", class_="result__a"):
        results.append(result.text.strip())

    return results

# Step 1: Input the URL
url = st.text_input("🔗 Enter webpage URL to scrape tables:", value="https://www.iitsystem.ac.in/mhrdprojects")

# Step 2: Ingest Tables
if st.button("🚀 Ingest and Preview Tables"):
    with st.spinner("Fetching and processing tables..."):
        try:
            tables, documents = fetch_tables_from_url(url)
            qa_chain, vectorstore = build_qa_chain(documents)

            st.session_state["qa_chain"] = qa_chain
            st.session_state["vectorstore"] = vectorstore
            st.session_state["tables"] = tables
            st.session_state["documents"] = documents

            st.success(f"✅ {len(tables)} tables ingested and ready for Q&A!")

        except Exception as e:
            st.error(f"❌ Failed to process URL: {str(e)}")

# Step 3: Show Tables and Q&A Input (if available)
if "tables" in st.session_state and "qa_chain" in st.session_state:
    for idx, (table, doc) in enumerate(zip(st.session_state["tables"], st.session_state["documents"])):
        title = doc.metadata["source"]

        with st.expander(f"📄 Preview: {title}"):
            st.dataframe(table)

            col1, col2 = st.columns(2)
            with col1:
                with open(os.path.join(DATA_DIR, f"{title}.csv"), "rb") as f:
                    st.download_button("⬇️ Download CSV", f, file_name=f"{title}.csv")

            with col2:
                with open(os.path.join(DATA_DIR, f"{title}.xlsx"), "rb") as f:
                    st.download_button("⬇️ Download XLSX", f, file_name=f"{title}.xlsx")

    st.markdown("---")
    st.subheader("💬 Ask a question about the data")

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

# Step 4: Internet Search with Summarization
st.markdown("---")
st.subheader("🌐 Internet Search")

search_query = st.text_input("Type something to search on the internet:")

if st.button("🔍 Search Internet"):
    if search_query.strip():
        with st.spinner("Searching the internet..."):
            try:
                # Perform DuckDuckGo search
                results = ddg(search_query, max_results=10)
                st.subheader("🌍 Search Results:")
                snippets = ""

                for res in results:
                    title = res.get("title", "")
                    href = res.get("href", "")
                    body = res.get("body", "")
                    snippets += f"- {title}: {body}\n"
                    st.markdown(f"[{title}]({href})")

                # Generate summary using LLM
                if snippets:
                    summarizer_chain = summary_prompt | llm | output_parser
                    summary = summarizer_chain.invoke({"query": search_query, "snippets": snippets})
                    st.markdown("**📝 Summary:**")
                    st.write(summary)
                else:
                    st.warning("No snippets found to summarize.")
            except Exception as e:
                st.error(f"❌ Error during search: {str(e)}")
    else:
        st.warning("Please enter a search query.")
