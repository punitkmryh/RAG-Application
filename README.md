# RAG-Application - Scraping-URL-Tables

## Building a **LangChain-powered Table Extractor + LLM Q&A + Downloadable CSV/Excel Generator** deployed on **Streamlit Cloud**. 

Here's exactly what we’ll do:

---

### ✅ What This Project Does

1. 🔗 **Takes a user-specified URL** (with tables in HTML)
2. 🧠 Extracts all **HTML tables** using **BeautifulSoup + Pandas**
3. 💬 Chains the **tables with an LLM (via Groq)** for Q&A via RAG
4. 📂 Lets the user:
   - Ask specific queries about the table
   - Select table(s) to **download as CSV or Excel**
5. 🚀 Fully deployable on **Streamlit Cloud**
6. 💾 Uses **FAISS + HuggingFace Embeddings** for vector storage

---

## 📁 Project Structure

```
table-extractor-rag/
│
├── app.py                    # Streamlit app
├── requirements.txt
├── .env                     # Contains GROQ API Key
├── vectorstore/             # Stores FAISS index (after ingestion)
│   └── faiss_index/
```

---
