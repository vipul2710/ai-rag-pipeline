import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stTextInput > label {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .metric-card {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .success-message {
        padding: 1rem;
        background: #d4edda;
        border-radius: 5px;
        color: #155724;
        margin: 1rem 0;
    }
    .error-message {
        padding: 1rem;
        background: #f8d7da;
        border-radius: 5px;
        color: #721c24;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 AI Document Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Transform PDFs into conversational knowledge bases with production-grade RAG pipeline</p>', unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API key input
    api_key = os.getenv("OPENAI_API_KEY") or st.text_input(
        "OpenAI API Key:",
        type="password",
        help="Enter your OpenAI API key. Get one at platform.openai.com"
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your API key to continue")
    
    st.markdown("---")
    
    # Current status
    st.subheader("📊 Status")
    if st.session_state.vectorstore:
        st.success(f"✅ {len(st.session_state.uploaded_file_names)} documents loaded")
        st.info("Ready for queries")
        
        # Clear documents button
        if st.button("🗑️ Clear All Documents", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.uploaded_file_names = []
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("No documents loaded")
    
    st.markdown("---")
    
    # Tech stack info
    st.subheader("🛠️ Tech Stack")
    st.markdown("""
    - **LangChain** - RAG framework
    - **ChromaDB** - Vector database
    - **OpenAI GPT-4** - Language model
    - **HuggingFace** - Embeddings
    - **Streamlit** - Web interface
    """)
    
    st.markdown("---")
    
    # Contact info
    st.subheader("👨‍💻 Built by Vipul")
    st.markdown("""
    **Data Engineer & Analytics Lead | 9 years exp**
    
    [LinkedIn](https://www.linkedin.com/in/vipulmeh/) | [Email](mailto:vipul@northstar.com)
    
    **Available for:**
    - ✅ RAG pipeline development
    - ✅ AI integration projects
    - ✅ Data engineering consulting
    - ✅ Technical architecture
    """)

# Check if API key is provided
if not api_key:
    st.info("👈 Please enter your OpenAI API key in the sidebar to get started")
    
    # Show what this can do
    st.markdown("---")
    st.subheader("✨ What You Can Build With This")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📚 Knowledge Bases")
        st.markdown("""
        - Company policies
        - Product documentation
        - Training materials
        - Employee handbooks
        - Standard procedures
        """)
    
    with col2:
        st.markdown("### 🤖 Customer Support")
        st.markdown("""
        - FAQ automation
        - Ticket resolution
        - Help desk bots
        - 24/7 assistance
        - Multi-language support
        """)
    
    with col3:
        st.markdown("### ⚖️ Document Analysis")
        st.markdown("""
        - Contract review
        - Compliance checking
        - Research synthesis
        - Due diligence
        - Audit support
        """)
    
    st.markdown("---")
    st.subheader("🎯 Perfect For AI Agencies")
    st.markdown("""
    - **White-label ready** - Brand it as your own
    - **Production-grade** - Battle-tested architecture
    - **Customizable** - Modify to client needs
    - **Scalable** - Handle 1000s of documents
    - **API-ready** - Integrate anywhere
    """)
    
    st.stop()

# Initialize embeddings (cached for performance)
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

try:
    embeddings = get_embeddings()
except Exception as e:
    st.error(f"Error loading embeddings model: {str(e)}")
    st.stop()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📄 Upload Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files (Max 10MB each)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF documents to analyze"
    )
    
    if uploaded_files:
        # Validate file sizes
        valid_files = []
        for file in uploaded_files:
            file_size_mb = file.size / (1024 * 1024)
            if file_size_mb > 10:
                st.error(f"❌ {file.name} is too large ({file_size_mb:.1f}MB). Maximum size is 10MB.")
            else:
                valid_files.append(file)
        
        if valid_files:
            # Process documents
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                try:
                    with st.spinner("📄 Processing documents... This may take a moment."):
                        all_docs = []
                        
                        # Progress bar for multiple files
                        if len(valid_files) > 1:
                            progress_bar = st.progress(0)
                        
                        for idx, uploaded_file in enumerate(valid_files):
                            # Save to temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_path = tmp_file.name
                            
                            try:
                                # Load PDF
                                loader = PyPDFLoader(tmp_path)
                                docs = loader.load()
                                
                                # Add source filename to metadata
                                for doc in docs:
                                    doc.metadata['source'] = uploaded_file.name
                                
                                all_docs.extend(docs)
                                
                            except Exception as e:
                                st.warning(f"⚠️ Could not process {uploaded_file.name}: {str(e)}")
                            
                            finally:
                                # Clean up temp file
                                os.unlink(tmp_path)
                            
                            # Update progress
                            if len(valid_files) > 1:
                                progress_bar.progress((idx + 1) / len(valid_files))
                        
                        if not all_docs:
                            st.error("❌ No documents could be processed. Please check your PDF files.")
                            st.stop()
                        
                        # Split text into chunks
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            length_function=len,
                        )
                        chunks = text_splitter.split_documents(all_docs)
                        
                        # Create vector store
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            collection_name="rag_demo_collection"
                        )
                        
                        # Store file names
                        st.session_state.uploaded_file_names = [f.name for f in valid_files]
                        
                        st.success(f"✅ Successfully processed {len(valid_files)} documents into {len(chunks)} searchable chunks!")
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
                    st.info("Please try with different PDFs or check if files are corrupted.")

with col2:
    if st.session_state.vectorstore:
        st.subheader("📊 Document Stats")
        
        # Display metrics in cards
        st.markdown(f"""
        <div class="metric-card">
            <strong>📁 Documents:</strong> {len(st.session_state.uploaded_file_names)}<br>
            <strong>📄 Files:</strong><br>
            {'<br>'.join(['• ' + name for name in st.session_state.uploaded_file_names])}
        </div>
        """, unsafe_allow_html=True)

# Query interface
if st.session_state.vectorstore:
    st.markdown("---")
    st.subheader("💬 Ask Questions About Your Documents")
    
    # Example questions
    with st.expander("💡 Example Questions", expanded=False):
        st.markdown("""
        - What are the main topics covered in these documents?
        - Summarize the key findings
        - What recommendations are mentioned?
        - List all dates and events mentioned
        - What are the policy requirements?
        - Who are the stakeholders mentioned?
        """)
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the key findings in these documents?",
        key="query_input"
    )
    
    # Search button
    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if query and search_button:
        try:
            with st.spinner("🔍 Searching through documents..."):
                
                # Search for relevant chunks
                relevant_docs = st.session_state.vectorstore.similarity_search(query, k=3)
                
                if not relevant_docs:
                    st.warning("⚠️ No relevant information found in the documents for this query.")
                else:
                    # Prepare context from relevant chunks
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    # Get AI response
                    client = OpenAI(api_key=api_key)
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        max_tokens=1024,
                        temperature=0.3,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that answers questions based solely on the provided context. If the answer is not in the context, say so clearly. Always be precise and cite specific information from the context."
                            },
                            {
                                "role": "user",
                                "content": f"""Based on the following context from documents, answer the question.

Context:
{context}

Question: {query}

Provide a clear, concise answer based only on the information in the context. If the answer isn't in the context, explicitly say "I cannot find this information in the provided documents." """
                            }
                        ]
                    )
                    
                    answer = response.choices[0].message.content
                    
                    # Display results
                    st.markdown("### 🎯 Answer")
                    st.markdown(f"""
                    <div class="success-message">
                        {answer}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sources
                    with st.expander("📚 View Source Chunks", expanded=False):
                        for i, doc in enumerate(relevant_docs, 1):
                            st.markdown(f"**Source {i}: {doc.metadata.get('source', 'Unknown')}**")
                            st.markdown(f"*Page: {doc.metadata.get('page', 'N/A')}*")
                            st.text_area(
                                f"Content {i}",
                                doc.page_content,
                                height=150,
                                key=f"source_{i}",
                                label_visibility="collapsed"
                            )
                            if i < len(relevant_docs):
                                st.markdown("---")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': query,
                        'answer': answer
                    })
                    
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.info("Please check your API key and try again.")
    
    # Show chat history
    if st.session_state.chat_history:
        with st.expander("📜 Previous Questions", expanded=False):
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                st.markdown(f"**Q{i}:** {chat['question']}")
                st.markdown(f"**A{i}:** {chat['answer']}")
                st.markdown("---")

else:
    # Instructions when no documents are loaded
    st.info("👆 Upload PDF documents above to get started")
    
    st.markdown("---")
    st.subheader("🚀 How It Works")
    
    col1, col2, col3,col4 = st.columns(4)
    
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.markdown("""
        Upload your PDF documents. The system will:
        - Extract all text
        - Preserve structure
        - Handle multiple files
        - Process in seconds
        """)
    
    with col2:
        st.markdown("### 2️⃣ Process")
        st.markdown("""
        Advanced processing:
        - Intelligent chunking
        - Semantic embeddings
        - Vector indexing
        - Metadata preservation
        """)
    
    with col3:
        st.markdown("### 3️⃣ Query")
        st.markdown("""
        Ask questions naturally:
        - Semantic search
        - Context-aware answers
        - Source citations
        - No hallucinations
        """)

    with col4:
        st.markdown("### ⚠️ Current Limitations")
        st.info("""
        **Note:** This demo version works best with text-based PDFs. 
        Tables and complex layouts may need additional processing.

        **Coming soon:**
        - Enhanced table extraction
        - Image text recognition (OCR)
        - Multi-column layout handling
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Built with ❤️ using LangChain, ChromaDB, OpenAI & Streamlit</p>
    <p style='font-size: 0.9rem;'>Production-grade RAG pipeline for AI agencies and enterprises</p>
</div>
""", unsafe_allow_html=True)