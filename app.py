import streamlit as st
import os
import hashlib
import pandas as pd
from dotenv import load_dotenv

# Loaders and RAG
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# Agent and Tools
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_experimental.agents import create_pandas_dataframe_agent

# Load environment variables
load_dotenv(override=True)
# UI Configuration
st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")
st.title("🤖 Agentic RAG Assistant")

# -------------------------------------------------------------
# 1. Configuration Setup
# -------------------------------------------------------------
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Initialize session state for file paths if not present
if "doc_paths" not in st.session_state:
    st.session_state.doc_paths = []
if "data_path" not in st.session_state:
    st.session_state.data_path = None
if "reload_trigger" not in st.session_state:
    st.session_state.reload_trigger = 0
@st.cache_resource
def initialize_rag_system(doc_paths, data_path, reload_trigger):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 **Missing API Key**: Please add your GOOGLE_API_KEY to the .env file or Streamlit Secrets.")
        return None, None, None, 0

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=0,
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    # Optimized for large PDFs: smaller chunks to avoid memory bottlenecks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )

    def get_file_hash(file_path):
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    # Consistently manage indexes within the uploads directory
    INDEX_DIR = os.path.join(UPLOAD_DIR, "faiss_indexes")
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)

    vectorstores = []
    total_chunks = 0
    existing_files = [p for p in doc_paths if os.path.exists(p)]

    for path in existing_files:
        try:
            file_hash = get_file_hash(path)
            abs_index_path = os.path.join(INDEX_DIR, f"faiss_{file_hash}")

            if os.path.exists(abs_index_path):
                vs = FAISS.load_local(
                    abs_index_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                # Count chunks for visibility
                total_chunks += vs.index.ntotal
                vectorstores.append(vs)
            else:
                ext = path.split(".")[-1].lower()
                if ext == "pdf":
                    loader = PyPDFLoader(path)
                elif ext == "txt":
                    loader = TextLoader(path)
                elif ext == "docx":
                    loader = Docx2txtLoader(path)
                else:
                    continue

                docs = loader.load()
                if not docs:
                    st.warning(f"⚠️ No content extracted from {os.path.basename(path)}")
                    continue

                for doc in docs:
                    doc.metadata["source"] = path

                chunks = text_splitter.split_documents(docs)
                if chunks:
                    total_chunks += len(chunks)
                    vs = FAISS.from_documents(chunks, embeddings)
                    vs.save_local(abs_index_path)
                    vectorstores.append(vs)
                else:
                    st.warning(f"⚠️ Could not split information from {os.path.basename(path)}")
        except Exception as e:
            error_msg = str(e)
            if "getaddrinfo failed" in error_msg:
                st.error("🌐 **Network Error**: Unable to reach Google AI services. Please check your internet connection and DNS settings.")
            else:
                st.error(f"❌ Error processing {os.path.basename(path)}: {error_msg}")

    vectorstore = None
    if vectorstores:
        vectorstore = vectorstores[0]
        for vs in vectorstores[1:]:
            vectorstore.merge_from(vs)

    pandas_agent = None
    if data_path and os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_excel(data_path)
            pandas_agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                allow_dangerous_code=True
            )
        except Exception as e:
            st.error(f"❌ Error loading data file: {str(e)}")

    return llm, vectorstore, pandas_agent, total_chunks

# Sidebar - Files Management
with st.sidebar:
    st.header("📁 Data Management")
    
    # 1. Upload Documents
    uploaded_docs = st.file_uploader(
        "Upload Documents (PDF, TXT, DOCX)", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True
    )
    
    # 2. Upload Data
    uploaded_data = st.file_uploader(
        "Upload Structured Data (CSV, XLSX)", 
        type=["csv", "xlsx"]
    )

    # 3. Trigger reload if files changed
    if st.button("🔄 Sync & Process Files"):
        with st.spinner("Saving and indexing your files..."):
            # Process Documents
            new_doc_paths = []
            if uploaded_docs:
                for uploaded_file in uploaded_docs:
                    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    new_doc_paths.append(save_path)
            st.session_state.doc_paths = new_doc_paths
            
            # Process Data
            if uploaded_data:
                save_path_data = os.path.join(UPLOAD_DIR, uploaded_data.name)
                with open(save_path_data, "wb") as f:
                    f.write(uploaded_data.getbuffer())
                st.session_state.data_path = save_path_data
            else:
                st.session_state.data_path = None

            st.session_state.reload_trigger += 1
            st.cache_resource.clear() # Clear cache to force reload with new files
            st.success("Documents synced!")
            st.rerun()

    # 4. Display Active Knowledge Base
    if st.session_state.doc_paths or st.session_state.data_path:
        st.divider()
        st.markdown("### 🟢 Active Knowledge Base")
        if st.session_state.doc_paths:
            for doc in st.session_state.doc_paths:
                st.write(f"- 📄 {os.path.basename(doc)}")
            if "total_chunks" in locals():
                st.caption(f"Indexed {total_chunks} text blocks")
        if st.session_state.data_path:
            st.write(f"- 📊 {os.path.basename(st.session_state.data_path)}")

# Initialize the system
with st.spinner("Initializing AI Brain..."):
    llm, vectorstore, pandas_agent, total_chunks = initialize_rag_system(
        st.session_state.doc_paths, 
        st.session_state.data_path, 
        st.session_state.get("reload_trigger", 0)
    )

# Final Polish: Show system readiness
if st.session_state.doc_paths:
    if vectorstore:
        st.toast("✅ Vector Database Ready", icon="🧠")
    else:
        st.error("❌ Vector Database could not be initialized. Please check your files.")
elif not st.session_state.doc_paths and not st.session_state.data_path:
    st.info("👋 Welcome! Please upload your documents or data in the sidebar to get started.")

# -------------------------------------------------------------
# 2. Define the Agent "Tools"
# -------------------------------------------------------------

@tool
def search_docs(query: str) -> str:
    """
    Use this tool to search through unstructured text documents (like PDFs, TXT, DOCX) 
    for qualitative information and text retrieval.
    """
    if not vectorstore:
        return "ERROR: No vector database available. User must upload and sync documents first."
    
    try:
        results = vectorstore.similarity_search(query, k=5)
        if not results:
            return "No matching information found in the current documents. Try a broader search."
            
        formatted_result = []
        for doc in results:
            content = doc.page_content
            page = doc.metadata.get("page", "N/A")
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            formatted_result.append(f"Source: {source} | Page: {page}\n{content}")
        return "\n\n---\n\n".join(formatted_result)
    except Exception as e:
        return f"CRITICAL ERROR while searching documents: {str(e)}"

@tool
def analyze_data(query: str) -> str:
    """
    Use this tool to perform data analysis, math computations, counting rows, filtering, 
    or aggregations on structured tabular datasets (CSV and Excel formats).
    """
    if not pandas_agent:
        return "No structured data (CSV/Excel) was loaded."
    response = pandas_agent.invoke({"input": query})
    return response["output"] if isinstance(response, dict) else str(response)

# -------------------------------------------------------------
# 3. Streamlit Chat Interface
# -------------------------------------------------------------

# Main Router Agent
@st.cache_resource
def get_agent(_llm, doc_paths, data_path):
    doc_names = [os.path.basename(p) for p in doc_paths] if doc_paths else []
    data_name = os.path.basename(data_path) if data_path else "None"
    docs_summary = ", ".join(doc_names) if doc_names else "None"

    system_prompt = f"""
You are an AI assistant that answers questions using provided tools.

### ACTIVE KNOWLEDGE BASE:
- **Documents**: {docs_summary}
- **Structured Data**: {data_name}

Follow these rules:
1. Use `search_docs` for qualitative information from the documents listed above.
2. Use `analyze_data` for quantitative analysis or math on the structured data file listed above.
3. Always include the **Source File Name** and **Page Number** (if available) in your response.
4. If the user asks about a specific file, confirm if it is in your knowledge base above.
5. If multiple sources are used, mention all clearly.
"""
    return create_agent(
        model=_llm,
        tools=[search_docs, analyze_data],
        system_prompt=system_prompt
    )

agent = get_agent(
    llm, 
    tuple(st.session_state.doc_paths), 
    st.session_state.data_path
)

# Sidebar for clear history
with st.sidebar:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Quick Actions Section
st.subheader("⚡ Quick Actions")
col1, col2, col3, col4, col5 = st.columns(5)

query = None

with col1:
    if st.button("📄 Summarize"):
        query = "Summarize all uploaded documents and give a high-level overview."

with col2:
    if st.button("🔍 Key Info"):
        query = "Extract all important names, dates, and terms from the documents."

with col3:
    if st.button("🧑 Resume"):
        query = "Extract skills, experience, and projects from the resume pdf."

with col4:
    if st.button("🏫 School"):
        query = "Find school name, session, and fee structure from the documents."

with col5:
    if st.button("📊 Data"):
        query = "Analyze the structured data in the CSV and give me 3 key insights."

# Chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process input (from chat input or quick action buttons)
prompt = st.chat_input("Ask me anything about your documents or data...")
if query:
    prompt = query

if prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Convert session history to LangChain message format
                history_messages = []
                for m in st.session_state.messages:
                    if m["role"] == "user":
                        history_messages.append(("user", m["content"]))
                    else:
                        history_messages.append(("assistant", m["content"]))

                # Using current messages for the agent call
                response = agent.invoke({"messages": history_messages})
                
                # Extract clean text from response
                final_content = response["messages"][-1].content
                if isinstance(final_content, list):
                    final_answer = "\n".join([block["text"] for block in final_content if block.get("type") == "text"])
                else:
                    final_answer = final_content
                
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
                # Reset query so it doesn't trigger on next rerun if chat_input is used
                if query:
                    st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")