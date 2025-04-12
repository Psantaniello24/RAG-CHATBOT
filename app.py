import os
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import fitz  # PyMuPDF
from typing import List, Tuple

# Load environment variables
load_dotenv()

# Constants
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gpt-3.5-turbo"
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()

def process_document(file) -> Tuple[List[str], List[str]]:
    """Process uploaded document and return chunks and sources."""
    if file.name.endswith('.pdf'):
        # Read PDF content using PyMuPDF
        pdf_content = []
        pdf_stream = file.read()
        pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pdf_content.append(page.get_text())
        
        text = "\n".join(pdf_content)
        pdf_document.close()
    else:
        text = file.getvalue().decode("utf-8")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_text(text)
    sources = [f"{file.name}"] * len(chunks)
    return chunks, sources

def update_vector_store(chunks: List[str], sources: List[str]):
    """Create or update vector store with new documents."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    if st.session_state.vector_store is None:
        # Create new vector store if none exists
        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            metadatas=[{"source": source} for source in sources]
        )
        st.session_state.vector_store = vector_store
    else:
        # Add new documents to existing vector store
        st.session_state.vector_store.add_texts(
            texts=chunks,
            metadatas=[{"source": source} for source in sources]
        )

def create_chain(vector_store):
    """Create conversation chain with memory."""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    llm = ChatOpenAI(
        model_name=st.session_state.model_name,
        temperature=0.7
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        get_chat_history=lambda h: h  # Return raw chat history
    )
    return chain

def generate_response(query: str) -> str:
    """Generate response using the conversation chain or fallback to general assistant."""
    # If no documents are uploaded, use general assistant mode
    if not st.session_state.chain:
        llm = ChatOpenAI(
            model_name=st.session_state.model_name,
            temperature=0.7
        )
        response = llm.invoke(
            f"Act as a helpful assistant. Question: {query}"
        ).content
        return f"[General Assistant Mode] {response}"
    
    # Try to get response from documents
    result = st.session_state.chain({"question": query})
    response = result["answer"]
    
    # Check if the response indicates no relevant information found
    no_info_indicators = [
        "I don't have enough information",
        "I cannot find",
        "I don't have access",
        "no relevant information",
        "cannot answer",
        "don't have any specific information",
        "no information available"
    ]
    
    response_lower = response.lower()
    if any(indicator.lower() in response_lower for indicator in no_info_indicators):
        # Fallback to general assistant mode
        llm = ChatOpenAI(
            model_name=st.session_state.model_name,
            temperature=0.7
        )
        general_response = llm.invoke(
            f"Act as a helpful assistant. If you can answer this question based on general knowledge, please do so. Question: {query}"
        ).content
        response = f"[General Assistant Mode] {general_response}"
    else:
        # Add source citations for document-based responses
        sources = set()
        for doc in result["source_documents"]:
            sources.add(doc.metadata["source"])
        
        if sources:
            response += "\n\nSources: " + ", ".join(sources)
    
    return response

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("Settings")
        
        # Model selection
        st.session_state.model_name = st.radio(
            "Select Model",
            ["gpt-3.5-turbo", "gpt-4"],
            index=0 if st.session_state.model_name == "gpt-3.5-turbo" else 1
        )
        
        # Document upload
        st.title("Document Upload")
        uploaded_file = st.file_uploader(
            "Upload PDF or Text",
            type=["pdf", "txt"]
        )
        
        if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
            with st.spinner("Processing document..."):
                chunks, sources = process_document(uploaded_file)
                update_vector_store(chunks, sources)
                st.session_state.chain = create_chain(st.session_state.vector_store)
                st.session_state.uploaded_files.add(uploaded_file.name)
                st.success(f"Document '{uploaded_file.name}' processed successfully!")
        
        # Display loaded documents
        if st.session_state.uploaded_files:
            st.subheader("Loaded Documents")
            for doc in sorted(st.session_state.uploaded_files):
                st.write(f"📄 {doc}")
            st.info("You can ask questions about your documents or general questions. The assistant will try to use document information when relevant.")
        else:
            st.info("No documents uploaded. The assistant will answer based on its general knowledge.")
        
        # Clear all button
        if st.button("Clear All"):
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.chain = None
            st.session_state.uploaded_files = set()
            st.experimental_rerun()
    
    # Main chat area
    st.title("RAG Chatbot")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Message RAG Chatbot..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt)
                st.markdown(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 