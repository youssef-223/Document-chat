import streamlit as st
import torch
import logging
from pathlib import Path
import tempfile
from generation_pipeline import DocumentAssistant
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up page configuration
st.set_page_config(
    page_title="Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading
@st.cache_resource(show_spinner=False)
def load_assistant():
    try:
        with st.spinner('Loading Document Assistant...'):
            assistant = DocumentAssistant()
        return assistant, None
    except Exception as e:
        return None, str(e)

def save_uploaded_file(uploaded_file):
    """Save uploaded file temporarily and return the path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def sanitize_collection_name(name):
    """Sanitize collection name to ensure it's valid for ChromaDB"""
    # Replace spaces with underscores and remove special characters
    return re.sub(r'[^a-zA-Z0-9_]', '', name.replace(' ', '_'))

def main():
    st.title("Document Assistant 📚")
    
    # Initialize assistant
    assistant, error = load_assistant()
    
    if error:
        st.error(f"Failed to initialize: {error}")
        if st.button("Retry"):
            st.experimental_rerun()
        return
    
    if not assistant:
        st.warning("Assistant not initialized properly")
        return
    
    # Initialize session state
    if 'collections' not in st.session_state:
        st.session_state.collections = {}  # Changed to dict to store both display name and internal name
    if 'response' not in st.session_state:
        st.session_state.response = None
    if 'current_collection' not in st.session_state:
        st.session_state.current_collection = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    # Sidebar information
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        This Document Assistant helps you:
        1. Process various document formats (PDF, DOCX, TXT)
        2. Ask questions about the documents
        3. Get AI-powered responses with source context
        """)
        
        # Document upload section in sidebar
        st.markdown("### Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        # Automatically process the document when uploaded
        if uploaded_file is not None and not st.session_state.processing:
            try:
                st.session_state.processing = True
                with st.spinner("Processing document..."):
                    # Save uploaded file temporarily
                    temp_path = save_uploaded_file(uploaded_file)
                    
                    # Create sanitized collection name
                    display_name = Path(uploaded_file.name).stem
                    collection_name = sanitize_collection_name(display_name)
                    
                    # Process the document
                    assistant.process_documents(temp_path, collection_name)
                    
                    # Add collection to session state
                    st.session_state.collections[display_name] = collection_name
                    st.session_state.current_collection = display_name
                    
                    st.success(f"Document '{uploaded_file.name}' processed successfully!")
                    
                    # Clean up temporary file
                    Path(temp_path).unlink()
                    
                st.session_state.processing = False
                    
            except Exception as e:
                logger.error(f"Document processing error: {e}")
                st.error(f"Error processing document: {str(e)}")
                st.session_state.processing = False
                
        # Display processed documents in sidebar
        if st.session_state.collections:
            st.markdown("### Processed Documents")
            for display_name in st.session_state.collections:
                st.markdown(f"- {display_name}")
                

    # Query section in main area
    st.markdown("### Ask Questions")
    
    # Document selection
    if st.session_state.collections:
        display_names = list(st.session_state.collections.keys())
        selected_display_name = st.selectbox(
            "Select document to query:",
            options=display_names
        )
        
        if selected_display_name:
            st.session_state.current_collection = selected_display_name
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="What are the main points discussed in the document?"
        )
        
        # Create columns for button and checkbox
        col1, col2 = st.columns([1, 5])
        
        with col1:
            submit_button = st.button("Get Answer", key="submit")
        
        with col2:
            show_context = st.checkbox("Show source context", key="show_context")
        
        # Process query when submit button is clicked
        if submit_button and query.strip() and st.session_state.current_collection:
            try:
                with st.spinner("Processing your query..."):
                    # Get the internal collection name
                    collection_name = st.session_state.collections[st.session_state.current_collection]
                    
                    # Verify that the collection exists
                    collection_exists = False
                    try:
                        assistant.client.get_collection(name=collection_name)
                        collection_exists = True
                    except Exception as e:
                        logger.error(f"Collection not found: {e}")
                    
                    if not collection_exists:
                        st.error(f"Document data not found. The collection '{st.session_state.current_collection}' may have been deleted or corrupted.")
                        return
                    
                    # Clear GPU cache
                    torch.cuda.empty_cache()
                    
                    # Get response
                    response = assistant.query_document(
                        question=query,
                        collection_name=collection_name
                    )
                    
                    # Display response
                    st.markdown("### Answer")
                    st.write(response)
                    
                    # Show context if requested
                    if show_context:
                        st.markdown("### Source Context")
                        collection = assistant.client.get_collection(
                            name=collection_name
                        )
                        question_embedding = assistant.embed_text(query)
                        results = collection.query(
                            query_embeddings=[question_embedding.tolist()],
                            n_results=3
                        )
                        
                        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                            with st.expander(f"Source {i}"):
                                st.write(doc)
                                st.caption(f"Chunk ID: {metadata['chunk_id']}")
                    
            except Exception as e:
                logger.error(f"Query processing error: {e}")
                st.error(f"Error processing query: {str(e)}")
                torch.cuda.empty_cache()
    else:
        st.info("Please upload a document using the sidebar.")

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit, Transformers, and ChromaDB*")

if __name__ == "__main__":
    main()
