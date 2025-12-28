import os
import shutil
from dotenv import load_dotenv
import streamlit as st
from streamlit_modal import Modal
from llama_index.core.schema import ImageNode
from PIL import Image
import base64
from io import BytesIO
import torch
from retriever import MongoDBChecker, MongoDBCollectionManager, extract_file_filter, IntelligentRetriever

torch.classes.__path__ = []         # set to avoid error about torch when using streamlit

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
PERSIST_DIR = os.getenv('PERSIST_DIR')
INDEX_INFO_PATH = os.getenv('INDEX_INFO_PATH')
IMAGE_FOLDER = os.getenv('IMAGE_FOLDER')

def main():
    if 'retriever' not in st.session_state:
        st.session_state['retriever'] = None
    if 'db_update' not in st.session_state:
        st.session_state['db_update'] = False

    st.title("Intelligent Retriever")

    st.sidebar.title("Database Management")
    st.sidebar.subheader("Document List")
    st.sidebar.write("Click the button below to check the database or fresh the doc list:")
    if st.sidebar.button("Check"):
        db_checker = MongoDBChecker(MONGO_URI, DB_NAME)
        db_docs = db_checker.get_db_docs()
        st.session_state['db_docs'] = db_docs  
        st.session_state['confirm_delete'] = None
        if not db_docs:
            st.sidebar.warning('Database empty.')
    else:
        db_docs = st.session_state.get('db_docs', {})

    if db_docs:
        with st.sidebar:
            for collection, files in db_docs.items():
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"#### 📁 {collection}")
                with col2:
                    # Collection delete button
                    if st.button(f"🗑️", key=f"delcol_{collection}"):
                        st.session_state['confirm_delete'] = collection

                if len(files) > 0:
                    db_manager = MongoDBCollectionManager(
                        MONGO_URI, DB_NAME, collection, PERSIST_DIR, IMAGE_FOLDER, INDEX_INFO_PATH
                    )

                    for file in files:
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.write(f"📄 {file}")
                        with col2:
                            if st.button("❌", key=f"del_{collection}_{file}"):
                                msg = db_manager.delete_files([file])
                                st.session_state['db_update'] = True
                                st.success('Deleted')

    st.sidebar.markdown("---")  # separator


    st.sidebar.subheader("Upload File(s)")
    uploaded_files = st.sidebar.file_uploader("Select file(s) to upload", accept_multiple_files=True)
    if st.sidebar.button("Upload"):
        if uploaded_files:
            UPLOAD_FOLDER = "uploads"
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            file_paths = []
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(temp_path)
            db_manager = MongoDBCollectionManager(MONGO_URI, DB_NAME, COLLECTION_NAME, PERSIST_DIR, IMAGE_FOLDER, INDEX_INFO_PATH)
            msg = db_manager.upload_file(file_paths)
            shutil.rmtree(UPLOAD_FOLDER)
            st.session_state['db_update'] = True
            st.sidebar.success(msg)
        else:
            st.sidebar.warning("Please select file(s) to upload.")
    
    # --- Confirmation Modal for Collection ---
    if st.session_state.get('confirm_delete'):
        collection = st.session_state['confirm_delete']
        modal = Modal("⚠️ Confirm Collection Deletion", key="collection_deletion_modal")
        with modal.container():
            st.warning(f"Are you sure to delete the entire collection: {collection}?")
            colA, colB = st.columns(2)
            with colA:
                if st.button("✅ Confirm Delete", key="confirm_delete_button"):
                    db_manager = MongoDBCollectionManager(MONGO_URI, DB_NAME, collection, PERSIST_DIR, IMAGE_FOLDER, INDEX_INFO_PATH)  
                    msg = db_manager.delete_collection()
                    st.success(msg)
                    st.session_state['confirm_delete'] = None
                    st.session_state['db_update'] = True
                    st.rerun()
            with colB:
                if st.button("❌ Cancel", key="cancel_delete_button"):
                    st.session_state['confirm_delete'] = None
                    st.rerun()

    # --- Retriever ---
    st.header("Retriever")
    query = st.text_input("Enter the query:")
    if st.button("Retrieve"):
        file_filter = extract_file_filter(query)
        if not file_filter:
            if st.session_state['retriever'] and (not st.session_state['db_update']):
                retriever = st.session_state['retriever']
            else:
                retriever = IntelligentRetriever(MONGO_URI, DB_NAME, COLLECTION_NAME, PERSIST_DIR, INDEX_INFO_PATH)
                retriever.build_retriever(file_filter)
                st.session_state['retriever'] = retriever
                st.session_state['db_update'] = False
        else:
            retriever = IntelligentRetriever(MONGO_URI, DB_NAME, COLLECTION_NAME, PERSIST_DIR, INDEX_INFO_PATH)
            retriever.build_retriever(file_filter)
        if retriever==None:
            st.warning('No retriever created.')
        else:
            retrieved_nodes = retriever.nodes_retrieval(query)
            if type(retrieved_nodes)==str:
                st.warning('No related information. Please check whether there are relevant data in the collection.')
            else:
                retrieved_image = []
                for res_node in retrieved_nodes:
                    if isinstance(res_node.node, ImageNode):
                        retrieved_image.append(res_node.node)
                    else:
                        st.write(res_node)
                if len(retrieved_image)>0:
                    cols = st.columns(len(retrieved_image))
                    for col, img_node in zip(cols, retrieved_image):
                        img = Image.open(BytesIO(base64.b64decode(img_node.image)))
                        path = img_node.metadata["file_path"]
                        img_name = path.split('/')[-1].split('\\')[-1]
                        col.image(img, caption=img_name, use_container_width=True)

if __name__ == "__main__":
    main()
