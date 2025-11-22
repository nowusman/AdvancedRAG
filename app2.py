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
import weaviate
import weaviate.classes.config as wvcc
import weaviateRetriever
import weaviateCollectionManager

torch.classes.__path__ = []         # set to avoid error about torch when using streamlit

load_dotenv()

COLLECTION_NAME = "Documents_llama"
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
        db_docs = weaviateCollectionManager.display_uploaded_files(COLLECTION_NAME)
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

                    for file in files:
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.write(f"📄 {file['file_name']}")
                        with col2:
                            if st.button("❌", key=f"del_{collection}_{file}"):
                                # msg = db_manager.delete_files([file])
                                msg = weaviateCollectionManager.delete_file(collection, file['file_path'])
                                st.session_state['db_update'] = True
                                st.success('Deleted')

    # st.sidebar.markdown("---")  # separator

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

            print(file_paths)
            
            msg = weaviateRetriever.main_upload(COLLECTION_NAME, UPLOAD_FOLDER, file_paths)
            st.sidebar.success(msg)
            shutil.rmtree(UPLOAD_FOLDER)
            st.session_state['db_update'] = True

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

                    msg = weaviateCollectionManager.delete_collection(collection)
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
        file_filter = weaviateRetriever.extract_file_filter(query)
        print(f'File filter: {file_filter}')

        ### bm25 query
        response = weaviateRetriever.search_bm25(COLLECTION_NAME, query)
        # st.write(res)
        retrieved_image = []
        for o in response:
            st.write(o.properties)
            # print(o.properties)

            print(o.properties["_node_type"])
            if o.properties["_node_type"] == "ImageNode":
                  # retrieved_image.append(res_node.node.metadata["file_path"])
                retrieved_image.append(o.properties)
            else:
                pass
                # st.write(res_node)

        if len(retrieved_image)>0:
            import json
            # for obj in retrieved_image:
            cols = st.columns(len(retrieved_image))
            for col, obj in zip(cols, retrieved_image):

                info = json.loads(obj["_node_content"])

                # img = Image.open(path)
                img = Image.open(BytesIO(base64.b64decode(info["image"])))
                path = info["metadata"]["file_path"]
                print(path)
                img_name = path.split('/')[-1].split('\\')[-1]
                col.image(img, caption=img_name, use_container_width=True)

        ### bm25 query end
        ### openai query
        # response = weaviateRetriever.main_query(COLLECTION_NAME, query)
        # st.write(response)

        # retrieved_nodes = response.source_nodes
        # retrieved_image = []
        # for res_node in retrieved_nodes:
        #     if isinstance(res_node.node, ImageNode):
        #             # retrieved_image.append(res_node.node.metadata["file_path"])
        #         retrieved_image.append(res_node.node)
        #     else:
        #          pass
        #          # st.write(res_node)
        # if len(retrieved_image)>0:
        #     print(retrieved_image)
        #     cols = st.columns(len(retrieved_image))
        #     for col, img_node in zip(cols, retrieved_image):
        #         print('img_node--')
        #         print(img_node)

        #         # img = Image.open(path)
        #         img = Image.open(BytesIO(base64.b64decode(img_node.image)))
        #         path = img_node.metadata["file_path"]
        #         img_name = path.split('/')[-1].split('\\')[-1]
        #         col.image(img, caption=img_name, use_container_width=True)
        ### openai query end

if __name__ == "__main__":
    main()