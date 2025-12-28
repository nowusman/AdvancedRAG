"""
Shared UI components for Streamlit applications.

This module extracts common UI patterns used across app.py and app2.py
to eliminate code duplication and ensure consistent user experience.
"""

import streamlit as st
from streamlit_modal import Modal
from typing import Dict, List, Callable, Optional, Any
from PIL import Image
from io import BytesIO
import base64

from logging_config import setup_logging
from constants import (
    EMOJI_FOLDER,
    EMOJI_FILE,
    EMOJI_DELETE,
    EMOJI_DELETE_COLLECTION,
    EMOJI_WARNING,
    EMOJI_SUCCESS,
    MSG_DATABASE_EMPTY,
    MSG_UPLOAD_SELECT_FILES
)

logger = setup_logging(__name__)


def render_document_list(
    db_docs: Dict[str, List],
    on_file_delete: Callable[[str, str], str],
    on_collection_delete_request: Callable[[str], None]
) -> None:
    """
    Render the document list sidebar with delete buttons.
    
    Args:
        db_docs: Dictionary mapping collection names to lists of file names or file dicts
        on_file_delete: Callback function(collection, file) -> status message
        on_collection_delete_request: Callback function(collection) -> None to mark for deletion
    
    Examples:
        >>> def handle_file_delete(collection, filename):
        >>>     return f"Deleted {filename}"
        >>> 
        >>> def handle_collection_delete(collection):
        >>>     st.session_state['confirm_delete'] = collection
        >>>
        >>> render_document_list(db_docs, handle_file_delete, handle_collection_delete)
    """
    if not db_docs:
        return
    
    with st.sidebar:
        for collection, files in db_docs.items():
            col1, col2 = st.columns([5, 1])
            
            with col1:
                st.markdown(f"#### {EMOJI_FOLDER} {collection}")
            
            with col2:
                # Collection delete button
                if st.button(EMOJI_DELETE_COLLECTION, key=f"delcol_{collection}"):
                    on_collection_delete_request(collection)
            
            if len(files) > 0:
                for file in files:
                    # Handle both string filenames and dict file objects
                    if isinstance(file, dict):
                        file_name = file.get('file_name', str(file))
                        file_id = file.get('file_path', file_name)
                    else:
                        file_name = file
                        file_id = file
                    
                    col1, col2 = st.columns([5, 1])
                    
                    with col1:
                        st.write(f"{EMOJI_FILE} {file_name}")
                    
                    with col2:
                        if st.button(EMOJI_DELETE, key=f"del_{collection}_{file_id}"):
                            msg = on_file_delete(collection, file)
                            st.session_state['db_update'] = True
                            st.success('Deleted')


def render_file_uploader(
    on_upload: Callable[[List[str]], str],
    accept_multiple: bool = True
) -> None:
    """
    Render the file uploader component.
    
    Args:
        on_upload: Callback function(file_paths) -> status message
        accept_multiple: Whether to accept multiple files
    
    Examples:
        >>> def handle_upload(file_paths):
        >>>     # Process files
        >>>     return "Upload successful"
        >>>
        >>> render_file_uploader(handle_upload)
    """
    import os
    import shutil
    from constants import FILE_UPLOAD_FOLDER
    
    st.sidebar.subheader("Upload File(s)")
    uploaded_files = st.sidebar.file_uploader(
        "Select file(s) to upload",
        accept_multiple_files=accept_multiple
    )
    
    if st.sidebar.button("Upload"):
        if uploaded_files:
            # Create upload folder
            if not os.path.exists(FILE_UPLOAD_FOLDER):
                os.makedirs(FILE_UPLOAD_FOLDER)
            
            file_paths = []
            files_to_upload = uploaded_files if accept_multiple else [uploaded_files]
            
            # Save uploaded files temporarily
            for uploaded_file in files_to_upload:
                temp_path = os.path.join(FILE_UPLOAD_FOLDER, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(temp_path)
            
            try:
                # Call upload handler
                msg = on_upload(file_paths)
                st.session_state['db_update'] = True
                st.sidebar.success(msg)
            except Exception as e:
                logger.error(f"Upload failed: {e}", exc_info=True)
                st.sidebar.error(f"Upload failed: {str(e)}")
            finally:
                # Cleanup upload folder
                if os.path.exists(FILE_UPLOAD_FOLDER):
                    shutil.rmtree(FILE_UPLOAD_FOLDER)
        else:
            st.sidebar.warning(MSG_UPLOAD_SELECT_FILES)


def render_collection_deletion_modal(
    collection_name: str,
    on_confirm: Callable[[str], str]
) -> None:
    """
    Render a confirmation modal for collection deletion.
    
    Args:
        collection_name: Name of collection to delete
        on_confirm: Callback function(collection_name) -> status message
    
    Examples:
        >>> def handle_confirm_delete(collection):
        >>>     return f"Deleted {collection}"
        >>>
        >>> if st.session_state.get('confirm_delete'):
        >>>     render_collection_deletion_modal(
        >>>         st.session_state['confirm_delete'],
        >>>         handle_confirm_delete
        >>>     )
    """
    modal = Modal(f"{EMOJI_WARNING} Confirm Collection Deletion", key="collection_deletion_modal")
    
    with modal.container():
        st.warning(f"Are you sure to delete the entire collection: {collection_name}?")
        
        colA, colB = st.columns(2)
        
        with colA:
            if st.button(f"{EMOJI_SUCCESS} Confirm Delete", key="confirm_delete_button"):
                try:
                    msg = on_confirm(collection_name)
                    st.success(msg)
                    st.session_state['confirm_delete'] = None
                    st.session_state['db_update'] = True
                    st.rerun()
                except Exception as e:
                    logger.error(f"Collection deletion failed: {e}", exc_info=True)
                    st.error(f"Deletion failed: {str(e)}")
        
        with colB:
            if st.button(f"{EMOJI_DELETE} Cancel", key="cancel_delete_button"):
                st.session_state['confirm_delete'] = None
                st.rerun()


def render_retrieval_results(
    retrieved_nodes: List[Any],
    no_results_message: str = "No related information found."
) -> None:
    """
    Render retrieval results including text nodes and images.
    
    Args:
        retrieved_nodes: List of retrieved nodes from the retriever
        no_results_message: Message to display when no results found
    
    Examples:
        >>> nodes = retriever.retrieve("What is machine learning?")
        >>> render_retrieval_results(nodes)
    """
    from llama_index.core.schema import ImageNode
    
    if isinstance(retrieved_nodes, str):
        st.warning(retrieved_nodes)
        return
    
    if not retrieved_nodes:
        st.warning(no_results_message)
        return
    
    # Separate text and image nodes
    text_nodes = []
    image_nodes = []
    
    for res_node in retrieved_nodes:
        if isinstance(res_node.node, ImageNode):
            image_nodes.append(res_node.node)
        else:
            text_nodes.append(res_node)
    
    # Display text results
    for node in text_nodes:
        st.write(node)
    
    # Display images in columns
    if image_nodes:
        cols = st.columns(len(image_nodes))
        for col, img_node in zip(cols, image_nodes):
            try:
                # Decode base64 image
                img = Image.open(BytesIO(base64.b64decode(img_node.image)))
                
                # Get image name from file path
                path = img_node.metadata.get("file_path", "")
                img_name = path.split('/')[-1].split('\\')[-1] if path else "Image"
                
                col.image(img, caption=img_name, use_container_width=True)
            except Exception as e:
                logger.error(f"Failed to display image: {e}")
                col.error(f"Failed to display image: {img_name}")


def render_properties_results(
    response_objects: List[Any],
    no_results_message: str = "No related information found."
) -> None:
    """
    Render Weaviate response objects with properties.
    
    This is specifically for Weaviate responses that return objects with properties.
    
    Args:
        response_objects: List of Weaviate response objects
        no_results_message: Message to display when no results found
    
    Examples:
        >>> response = weaviate_client.query(...)
        >>> render_properties_results(response.objects)
    """
    import json
    
    if not response_objects:
        st.warning(no_results_message)
        return
    
    # Separate text and image results
    text_results = []
    image_results = []
    
    for obj in response_objects:
        properties = obj.properties
        
        # Display text properties
        st.write(properties)
        
        # Check for image nodes
        if properties.get("_node_type") == "ImageNode":
            image_results.append(properties)
    
    # Display images in columns
    if image_results:
        cols = st.columns(len(image_results))
        for col, obj in zip(cols, image_results):
            try:
                # Parse node content
                info = json.loads(obj["_node_content"])
                
                # Decode base64 image
                img = Image.open(BytesIO(base64.b64decode(info["image"])))
                
                # Get image name
                path = info["metadata"].get("file_path", "")
                img_name = path.split('/')[-1].split('\\')[-1] if path else "Image"
                
                col.image(img, caption=img_name, use_container_width=True)
            except Exception as e:
                logger.error(f"Failed to display image: {e}")
                col.error("Failed to display image")


def initialize_session_state(defaults: Optional[Dict[str, Any]] = None) -> None:
    """
    Initialize session state variables with defaults.
    
    Args:
        defaults: Dictionary of default values for session state
    
    Examples:
        >>> initialize_session_state({
        >>>     'retriever': None,
        >>>     'db_update': False,
        >>>     'db_docs': {}
        >>> })
    """
    default_values = {
        'retriever': None,
        'db_update': False,
        'db_docs': {},
        'confirm_delete': None
    }
    
    if defaults:
        default_values.update(defaults)
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar_header(title: str = "Database Management") -> None:
    """
    Render consistent sidebar header.
    
    Args:
        title: Title text for the sidebar
    
    Examples:
        >>> render_sidebar_header("Document Management")
    """
    st.sidebar.title(title)
    st.sidebar.subheader("Document List")
    st.sidebar.write("Click the button below to check the database or refresh the doc list:")


def render_query_interface(
    on_retrieve: Callable[[str], Any],
    header: str = "Retriever",
    placeholder: str = "Enter the query:"
) -> None:
    """
    Render the query interface with input and button.
    
    Args:
        on_retrieve: Callback function(query) -> results
        header: Header text for the query section
        placeholder: Placeholder text for the input
    
    Examples:
        >>> def handle_query(query_text):
        >>>     return retriever.retrieve(query_text)
        >>>
        >>> render_query_interface(handle_query)
    """
    st.header(header)
    query = st.text_input(placeholder)
    
    if st.button("Retrieve"):
        if query:
            try:
                results = on_retrieve(query)
                return results
            except Exception as e:
                logger.error(f"Query failed: {e}", exc_info=True)
                st.error(f"Query failed: {str(e)}")
                return None
        else:
            st.warning("Please enter a query.")
            return None

