import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Page Configuration
st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📚", layout="wide")
st.title("📚 Minamoto Intelligence")
st.caption("🚀 Intelligent PDF Analysis & Interactive Assistant")

# Embedding Model Caching
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = get_embedding_model()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper Function to Process PDF ---
def process_pdf(file_obj):
    with st.spinner("Processing PDF and updating database..."):
        file_obj.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_obj.read())
            tmp_file_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)

            if not chunks:
                st.error("No text could be extracted from this PDF.")
            else:
                Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="chroma_db",
                )
                st.success(f"Successfully processed {len(chunks)} text chunks!")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# --- Sidebar: History Controls & Alternative Upload ---
with st.sidebar:
    st.header("Settings & Options")
    
    # 1. Clear Chat History Option
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("Document Upload")
    sidebar_file = st.file_uploader("Upload PDF (Sidebar)", type=["pdf"], key="sidebar_pdf")
    if sidebar_file and st.button("Process Sidebar PDF"):
        process_pdf(sidebar_file)

# Initialize Vectorstore & Retriever
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)

# LLM & Prompt
llm = ChatGroq(model="openai/gpt-oss-120b")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
      Use ONLY the provided context to answer the question. 
    If the answer is not present in the context,
      say: "I couldn't find the answer in the document." """),
    ("human", """Context:{context} Question:{question}"""),
])

# --- Main Interface: Display Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Main Input Area (Mobile / Small Window Friendly Upload) ---
# Compact Layout for PDF upload right above or near chat input
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        main_uploaded_file = st.file_uploader(
            "Attach PDF for processing", 
            type=["pdf"], 
            key="main_pdf", 
            label_visibility="collapsed"
        )
    with col2:
        if main_uploaded_file and st.button("Upload PDF", key="process_main_pdf"):
            process_pdf(main_uploaded_file)

# Chat Input Box
if user_query := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and thinking..."):
            docs = retriever.invoke(user_query)
            context = "\n\n".join([doc.page_content for doc in docs])

            final_prompt = prompt_template.invoke({
                "context": context,
                "question": user_query
            })

            response = llm.invoke(final_prompt)
            st.write(response.content)

    st.session_state.messages.append({"role": "assistant", "content": response.content})