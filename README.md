# 🤖 Agentic RAG Assistant

A powerful, multimodal AI assistant designed to interact with your local documents and datasets. This project combines **Retrieval-Augmented Generation (RAG)** for unstructured text and **Agentic Data Analysis** for structured tabular data.

Built with **Streamlit**, **LangChain**, and **Google Gemini**.

---

## 🚀 Features

- **Unstructured Search (RAG)**: Upload PDFs, TXT, or DOCX files. The agent chunks and indexes them using **FAISS** and Google's latest embedding models.
- **Structured Data Analysis**: Upload CSV or Excel files. A smart **Pandas Agent** can perform calculations, create summaries, and answer questions about your data.
- **Intelligent Routing**: The agent automatically decides whether to search through documents or analyze data based on your query.
- **Manual Sync & Refresh**: Complete control over when files are processed. Sync only when you're ready.
- **Quick Actions**: One-click buttons for common tasks like summarizing documents, extracting key info, or analyzing resumes.
- **Persistent Memory**: Maintains chat history for context-aware conversations.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Orchestration**: [LangChain](https://www.langchain.com/)
- **Embeddings**: `GoogleGenerativeAIEmbeddings`
- **Vector Database**: `FAISS`
- **Models**: [Google Gemini 2.5 Flash](https://aistudio.google.com/)
- **Data Handling**: `Pandas`, `RecursiveCharacterTextSplitter`, `PyPDFLoader`

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rajrounak21/Agentic-Rag-Agent.git
   cd Agentic-Agent-Rag
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**:
   Create a `.env` file in the root directory and add your Google API Key:
   ```env
   GOOGLE_API_KEY=your_gen_ai_api_key_here
   ```

---

## 🎮 How to Use

1. **Run the App**:
   ```bash
   streamlit run app.py
   ```

2. **Upload Files**:
   Open the sidebar and use the file uploaders to select your documents (PDF/TXT/DOCX) or spreadsheets (CSV/XLSX).

3. **Sync Documents**:
   Click the **"🔄 Sync & Process Files"** button. This will process the files and prepare the knowledge base.

4. **Chat**:
   Ask questions in the chat box or use the **⚡ Quick Action** buttons to get immediate results!

---

## 📁 Directory Structure

```text
├── app.py              # Main Streamlit application
├── uploads/            # Temporary directory for uploaded files
├── faiss_*/            # Local FAISS indices for processed files
├── .env                # API Key configuration
└── requirements.txt    # Project dependencies
```

---

## 📜 License

MIT License - feel free to use and modify for your own projects!
