import streamlit as st
import asyncio
import os
import tempfile
import threading
import queue
from pathlib import Path
from typing import List
import time
from client import RAGMultiDocumentAgent
from config import Config
import logging
import traceback
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Large File Document Processor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}

class AsyncRunner:
    """Helper class to run async functions in Streamlit safely"""
    
    @staticmethod
    def run_in_thread(async_func, *args, timeout_minutes=15, **kwargs):  # Increased from 5 to 15
        """Run async function in a separate thread with proper event loop"""
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def run_async():
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Run the async function
                result = loop.run_until_complete(async_func(*args, **kwargs))
                result_queue.put(result)
                
            except Exception as e:
                logger.error(f"Async function error: {e}")
                logger.error(traceback.format_exc())
                error_queue.put(e)
            finally:
                try:
                    loop.close()
                except:
                    pass
        
        # Start thread
        thread = threading.Thread(target=run_async)
        thread.start()
        thread.join(timeout=timeout_minutes * 60)  # Convert to seconds
        
        # Check results
        if not error_queue.empty():
            raise error_queue.get()
        
        if not result_queue.empty():
            return result_queue.get()
        
        if thread.is_alive():
            raise TimeoutError(f"Operation timed out after {timeout_minutes} minutes")
        
        raise RuntimeError("Async operation failed without error")

def check_documents_in_store(agent):
    """Check if documents exist in the RAG store"""
    try:
        result = AsyncRunner.run_in_thread(agent.get_document_summary, timeout_minutes=1)
        if result and 'messages' in result:
            # Parse the summary to check document count
            summary_text = result['messages'][-1].content
            return "total_documents" in summary_text.lower() and "0" not in summary_text
        return False
    except Exception as e:
        logger.error(f"Error checking document store: {e}")
        return False

def initialize_agent():
    """Initialize the RAG agent with error handling"""
    try:
        if not Config.API_KEY:
            if Config.USE_AZURE_OPENAI:
                st.error("⚠️ Azure OpenAI API key not found. Please set the AZURE_OPENAI_API_KEY environment variable.")
            else:
                st.error("⚠️ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            return None
        
        if st.session_state.agent is None:
            with st.spinner("Initializing RAG agent..."):
                try:
                    st.session_state.agent = RAGMultiDocumentAgent()
                    st.success("✅ RAG agent initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize agent: {e}")
                    logger.error(f"Agent initialization error: {e}")
                    logger.error(traceback.format_exc())
                    return None
        
        return st.session_state.agent
    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {e}")
        logger.error(f"Agent initialization error: {e}")
        return None

def save_uploaded_files(uploaded_files):
    """Save uploaded files to temporary directory with validation"""
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        try:
            # Validate file size
            file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
            if file_size_mb > Config.MAX_FILE_SIZE_MB:
                st.error(f"❌ File {uploaded_file.name} is too large ({file_size_mb:.1f}MB). Max size: {Config.MAX_FILE_SIZE_MB}MB")
                continue
            
            # Create temp file
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            saved_paths.append(file_path)
            
            # Store file info
            st.session_state.processing_status[uploaded_file.name] = {
                "path": file_path,
                "size_mb": file_size_mb,
                "status": "uploaded"
            }
            
            logger.info(f"Saved file: {uploaded_file.name} ({file_size_mb:.1f}MB)")
            
        except Exception as e:
            st.error(f"❌ Failed to save {uploaded_file.name}: {e}")
            logger.error(f"File save error: {e}")
    
    return saved_paths

def process_documents_safely(agent, file_paths):
    """Process documents with comprehensive error handling"""
    try:
        return AsyncRunner.run_in_thread(agent.upload_and_index_documents, file_paths)
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        logger.error(traceback.format_exc())
        # raise

def query_documents_safely(agent, query):
    """Query documents with error handling"""
    try:
        return AsyncRunner.run_in_thread(agent.rag_query, query)
    except Exception as e:
        logger.error(f"Query error: {e}")
        logger.error(traceback.format_exc())
        raise

def analyze_collection_safely(agent):
    """Analyze collection with error handling"""
    try:
        return AsyncRunner.run_in_thread(agent.analyze_document_collection)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        logger.error(traceback.format_exc())
        raise

def research_safely(agent, research_question):
    """Perform research with error handling"""
    try:
        return AsyncRunner.run_in_thread(agent.multi_document_research, research_question)
    except Exception as e:
        logger.error(f"Research error: {e}")
        logger.error(traceback.format_exc())
        raise

# Main app
def main():
    st.markdown('<h1 class="main-header">📚 RAG Large File Document Processor</h1>', unsafe_allow_html=True)
    
    # Debug information
    with st.expander("🔧 Debug Information"):
        st.write(f"Python version: {sys.version}")
        st.write(f"Asyncio policy: {asyncio.get_event_loop_policy()}")
        st.write(f"Config loaded: {bool(Config.API_KEY)}")
        st.write(f"Agent initialized: {st.session_state.agent is not None}")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key status
        if Config.API_KEY:
            if Config.USE_AZURE_OPENAI:
                st.markdown('<div class="success-box">✅ Azure OpenAI API Key configured</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ OpenAI API Key configured</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">❌ API Key not found</div>', unsafe_allow_html=True)
            if Config.USE_AZURE_OPENAI:
                st.info("Set AZURE_OPENAI_API_KEY environment variable")
            else:
                st.info("Set OPENAI_API_KEY environment variable")
        
        # Model settings
        if Config.USE_AZURE_OPENAI:
            available_models = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"]
            default_model = Config.AZURE_OPENAI_DEPLOYMENT_NAME or "gpt-4.1-mini"
        else:
            available_models = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"]
            default_model = "gpt-4.1"
        
        # Find index of default model or use 0
        try:
            default_index = available_models.index(default_model)
        except ValueError:
            default_index = 0
            
        model_name = st.selectbox(
            "Model",
            available_models,
            index=default_index
        )
        
        # File size limits
        st.info(f"📊 Max file size: {Config.MAX_FILE_SIZE_MB}MB")
        st.info(f"🧠 Memory threshold: {Config.MEMORY_THRESHOLD_MB}MB")
        st.info(f"📝 Chunk size: {Config.CHUNK_SIZE} chars")
        
        # Clear documents button
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.session_state.agent:
                with st.spinner("Clearing documents..."):
                    try:
                        result = AsyncRunner.run_in_thread(st.session_state.agent.clear_documents)
                        st.success("✅ All documents cleared!")
                        st.session_state.uploaded_files = []
                        st.session_state.processing_status = {}
                    except Exception as e:
                        st.error(f"❌ Error clearing documents: {e}")
                        logger.error(f"Clear documents error: {e}")
    
    # Initialize agent
    agent = initialize_agent()
    if not agent:
        st.stop()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Documents", "🔍 Query Documents", "📊 Analyze Collection", "🔬 Research"])
    
    with tab1:
        st.header("📤 Upload and Process Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'docx', 'xlsx', 'xls', 'csv', 'pptx', 'ppt', 'pptm', 'potx', 
                'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'gif', 'webp'],
            accept_multiple_files=True,
            help=f"Supported formats: PDF, DOCX, Excel, CSV, PowerPoint, Images (JPG, PNG, etc.). Max size: {Config.MAX_FILE_SIZE_MB}MB per file"
        )
        
        if uploaded_files:
            st.subheader("📋 File Summary")
            
            # Display file information
            total_size = 0
            large_files = 0
            
            for file in uploaded_files:
                size_mb = len(file.getbuffer()) / (1024 * 1024)
                total_size += size_mb
                
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.write(f"📄 {file.name}")
                with col2:
                    st.write(f"{size_mb:.1f} MB")
                with col3:
                    if size_mb > Config.MEMORY_THRESHOLD_MB:
                        st.write("🔄 Large")
                        large_files += 1
                    else:
                        st.write("⚡ Standard")
                with col4:
                    st.write(f"📊 {file.type}")
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", len(uploaded_files))
            with col2:
                st.metric("Total Size", f"{total_size:.1f} MB")
            with col3:
                st.metric("Large Files", large_files)
            
            # Process button
            # Process button
            if st.button("🚀 Process Documents", type="primary"):
                try:
                    # Save files
                    with st.spinner("Saving uploaded files..."):
                        file_paths = save_uploaded_files(uploaded_files)
                    
                    if not file_paths:
                        st.error("❌ No files were successfully saved")
                        return
                    
                    # Process documents with better progress tracking
                    progress_container = st.container()
                    with progress_container:
                        st.info("🔄 **Large files detected - processing may take 10-15 minutes**")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Show estimated time for large files
                        total_size = sum(len(f.getbuffer()) for f in uploaded_files) / (1024 * 1024)
                        if total_size > 30:  # If total size > 30MB
                            estimated_minutes = max(5, int(total_size / 10))  # Rough estimate
                            st.warning(f"⏱️ Estimated processing time: {estimated_minutes} minutes for {total_size:.1f}MB")
                        
                        try:
                            # Update progress
                            for i, file_path in enumerate(file_paths):
                                progress_bar.progress((i + 1) / len(file_paths) * 0.3)
                                status_text.text(f"Preparing {os.path.basename(file_path)}...")
                                time.sleep(0.1)
                            
                            # Process all files with longer timeout
                            status_text.text("🔄 Processing documents with RAG (this may take 10-15 minutes for large files)...")
                            progress_bar.progress(0.5)
                            
                            # Use longer timeout for large files
                            timeout_mins = 20 if total_size > 30 else 10
                            result = AsyncRunner.run_in_thread(
                                agent.upload_and_index_documents, 
                                file_paths, 
                                timeout_minutes=timeout_mins
                            )
                            
                            progress_bar.progress(1.0)
                            status_text.text("✅ Processing complete!")
                            
                            # Display results and update session state
                            if result:
                                st.markdown('<div class="success-box">✅ Documents processed successfully!</div>', unsafe_allow_html=True)
                                
                                # Extract results from agent response
                                if 'messages' in result and result['messages']:
                                    last_message = result['messages'][-1].content
                                    st.text_area("Processing Results", last_message, height=200)
                                else:
                                    st.success("Documents processed successfully!")
                                
                                # Update session state
                                st.session_state.uploaded_files.extend([os.path.basename(p) for p in file_paths])
                                
                                # Force a rerun to update the UI
                                st.rerun()
                            
                        except TimeoutError as e:
                            progress_bar.progress(0.5)
                            status_text.text("⏱️ Processing is taking longer than expected...")
                            st.warning("""
                            🔄 **Processing is taking longer than expected but may still be running in the background.**
                            
                            **Options:**
                            1. Wait a few more minutes and try querying documents
                            2. Check the terminal logs for progress
                            3. Restart the application if needed
                            
                            Large PDFs with OCR can take 15-20 minutes to process.
                            """)
                            
                except Exception as e:
                            progress_bar.progress(0)
                            status_text.text("❌ Processing failed")
                            st.error(f"❌ Error processing documents: {e}")
                            
                            with st.expander("🔍 Error Details"):
                                st.code(traceback.format_exc())
    
    with tab2:
        st.header("🔍 Query Documents")
        
        # Check both session state and actual document store
        has_session_files = bool(st.session_state.uploaded_files)
        
        # Check if documents exist in the store (even if session state is empty)
        if not has_session_files:
            with st.spinner("Checking for existing documents..."):
                has_store_documents = check_documents_in_store(agent)
        else:
            has_store_documents = True
        
        if not has_session_files and not has_store_documents:
            st.markdown('<div class="info-box">ℹ️ Please upload documents first in the Upload tab.</div>', unsafe_allow_html=True)
        else:
            if has_session_files:
                st.markdown(f'<div class="success-box">📚 Ready to query {len(st.session_state.uploaded_files)} documents</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">📚 Documents found in store - ready to query!</div>', unsafe_allow_html=True)
                st.info("💡 Note: Documents were processed but session state was lost. Functionality is still available.")
            
            # Rest of your query logic...
            query = st.text_area(
                "Enter your question:",
                placeholder="e.g., What are the main trends in the sales data? How do customer feedback correlate with financial performance?",
                height=100
            )
        
        
            
            # Query examples
            with st.expander("💡 Example Questions"):
                example_queries = [
                    "What are the key findings across all documents?",
                    "Summarize the financial performance metrics",
                    "What trends appear in the data over time?",
                    "How do different documents relate to each other?",
                    "What are the main recommendations mentioned?",
                    "Extract all numerical data and key statistics",
                    "What issues or problems are identified?",
                    "Compare data between different time periods"
                ]
                
                for i, example in enumerate(example_queries):
                    if st.button(f"📝 {example}", key=f"example_{i}"):
                        query = example
            
            # Query button
            if st.button("🔍 Search", type="primary") and query.strip():
                with st.spinner("Searching documents using RAG..."):
                    try:
                        result = query_documents_safely(agent, query)
                        
                        if result and 'messages' in result:
                            last_message = result['messages'][-1].content
                            
                            st.subheader("🎯 Answer")
                            st.markdown(last_message)
                            
                            # Show raw result in expander
                            with st.expander("🔍 Detailed Results"):
                                st.json(result)
                    
                    except Exception as e:
                        st.error(f"❌ Query failed: {e}")
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())
                        logger.error(f"Query error: {e}")
    
    with tab3:
        st.header("📊 Analyze Document Collection")
        
        if not st.session_state.uploaded_files:
            st.markdown('<div class="info-box">ℹ️ Please upload documents first to analyze the collection.</div>', unsafe_allow_html=True)
        else:
            if st.button("📊 Analyze Collection", type="primary"):
                with st.spinner("Analyzing document collection..."):
                    try:
                        result = analyze_collection_safely(agent)
                        
                        if result and 'messages' in result:
                            last_message = result['messages'][-1].content
                            
                            st.subheader("📈 Collection Analysis")
                            st.markdown(last_message)
                            
                            # Show collection statistics
                            with st.expander("📋 Detailed Statistics"):
                                st.json(result)
                    
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())
                        logger.error(f"Analysis error: {e}")
    
    with tab4:
        st.header("🔬 Comprehensive Research")
        
        if not st.session_state.uploaded_files:
            st.markdown('<div class="info-box">ℹ️ Please upload documents first to perform research.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">🔬 Perform comprehensive research across all uploaded documents using multiple RAG queries and analysis.</div>', unsafe_allow_html=True)
            
            # Research question input
            research_question = st.text_area(
                "Research Question:",
                placeholder="e.g., Based on all available data, what strategic recommendations can be made for improving business performance?",
                height=100
            )
            
            # Research button
            if st.button("🔬 Conduct Research", type="primary") and research_question.strip():
                with st.spinner("Conducting comprehensive research... This may take several minutes."):
                    try:
                        result = research_safely(agent, research_question)
                        
                        if result and 'messages' in result:
                            last_message = result['messages'][-1].content
                            
                            st.subheader("📋 Research Report")
                            st.markdown(last_message)
                            
                            # Download button for research report
                            st.download_button(
                                label="📄 Download Research Report",
                                data=last_message,
                                file_name=f"research_report_{int(time.time())}.md",
                                mime="text/markdown"
                            )
                    
                    except Exception as e:
                        st.error(f"❌ Research failed: {e}")
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())
                        logger.error(f"Research error: {e}")

    # Footer
    st.markdown("---")
    st.markdown(
        "🤖 **MCP-RAG ** | "
        "Powered by MCP, LangChain | "
        f"Max file size: {Config.MAX_FILE_SIZE_MB}MB | "
        f"Supports: PDF, DOCX, Excel, CSV, PowerPoint, Images"

    )

if __name__ == "__main__":
    main()
