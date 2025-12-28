import os
import weaviate
import weaviate.classes.config as wvcc
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.core.node_parser import LangchainNodeParser, SimpleFileNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import ImageNode
from PIL import Image
import base64
import re
import requests
import torch
import openai
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
EMBED_MODEL_PATH = os.getenv('EMBED_MODEL_PATH')
IMAGE_FOLDER = os.getenv('IMAGE_FOLDER')

class WeaviateAutoRetriever:
    def __init__(self, embedding_model_path="./models/bge-base-en-v1.5"):
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.setup_openai_with_proxy()
        # Initialize LLM
        self.llm = OpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30.0,
            max_retries=2
        )
        Settings.llm = self.llm
        
        # connect to Weaviate
        self.client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051
        )
    
    def setup_openai_with_proxy(self):
        """Set OpenAI proxy"""
        proxy_url = os.environ.get('http_proxy') or os.environ.get('https_proxy')
        
        if proxy_url:
            logger.info("OpenAI proxy configured successfully")
            
            try:
                openai.proxy = proxy_url
            except AttributeError:
                logger.warning("Failed to set OpenAI proxy attribute")
            
        else:
            logger.debug("No proxy settings detected")

    def setup_vector_store(self, class_name="Documents_llama"):
        """Set Weaviate vector storage"""
        if not self.client.collections.exists(class_name):
            # Set collention
            self.client.collections.create(
                name=class_name,
                properties=[
                    wvcc.Property(
                        name="content",
                        data_type=wvcc.DataType.TEXT
                    ),
                    wvcc.Property(
                        name="file_path",
                        data_type=wvcc.DataType.TEXT
                    ),
                    wvcc.Property(
                        name="file_type",
                        data_type=wvcc.DataType.TEXT
                    ),
                    wvcc.Property(
                        name="page_number",
                        data_type=wvcc.DataType.NUMBER
                    )
                ],
                vectorizer_config=wvcc.Configure.Vectorizer.none()
            )
        
        return WeaviateVectorStore(
            weaviate_client=self.client,
            index_name=class_name
        )
    
    def load_and_process_documents(self, directory_path):
        """Loading and processing documents"""
        reader = SimpleDirectoryReader(
            input_dir=directory_path,
            recursive=True,
            required_exts=[".txt", ".docx", ".pdf"]
        )
        documents = reader.load_data()
        
        for doc in documents:
            file_path = doc.metadata.get('file_path', '')
            doc.metadata['file_type'] = os.path.splitext(file_path)[1].lower().replace('.', '')
            doc.metadata['page_number'] = doc.metadata.get('page_label', 0)
        
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        
        nodes = node_parser.get_nodes_from_documents(documents)
        return nodes
    
    def load_and_process_documents2(self, directory_path):
        """Loading and processing documents (text only)"""
        reader = SimpleDirectoryReader(
            input_dir=directory_path,
            recursive=True,
            required_exts=[".txt", ".docx"]
        )
        documents = reader.load_data()

        logger.debug("Processing documents from %s", directory_path)
        
        for doc in documents:
            file_path = doc.metadata.get('file_path', '')
            doc.metadata['file_type'] = os.path.splitext(file_path)[1].lower().replace('.', '')
            doc.metadata['page_number'] = doc.metadata.get('page_label', 0)
        
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        
        nodes = node_parser.get_nodes_from_documents(documents)
        logger.debug("Generated %d nodes from documents", len(nodes))
        return nodes
    
    
    def load_and_process_pdf(self, docling_paths):
        """Loading and processing pdf"""
        from llama_index.readers.docling import DoclingReader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from datetime import datetime
        docling_reader = DoclingReader()
        docling_docs = []
        for docling_path in docling_paths:
            docling_doc = docling_reader.load_data(docling_path)
            for doc in docling_doc:
                doc.metadata = {
                    "file_path": docling_path,
                    "file_name": docling_path.split('/')[-1].split('\\')[-1],
                    "creation_data": str(datetime.now().date()),
                    "last_modified_date": str(datetime.now().date()),
                }
            docling_docs += docling_doc

        parser = LangchainNodeParser(RecursiveCharacterTextSplitter(separators='#'))
        nodes = parser.get_nodes_from_documents(docling_docs)
        print('docs nodes--------')
        print(nodes)
        return nodes

    def load_and_process_images(self, directory_path, file_path, data_collection):
        """Loading and processing documents"""
        save_folder = os.path.join(IMAGE_FOLDER, data_collection)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        mixed_files = []
        for file in file_path:
            image_name = file.split('/')[-1].split('.')[0].split("\\")[-1]
            if not os.path.exists(f'{save_folder}/{file.split("/")[-1].split("\\")[-1]}'):
                image = Image.open(file)
                # image.save(f'{save_folder}/{file.split('/')[-1].split("\\")[-1]}')
                image.save(str(save_folder)+'/'+str(file.split('/')[-1].split("\\")[-1]))
            if not os.path.exists(f'{save_folder}/{image_name}.txt'):
                image_doc = SimpleDirectoryReader(input_files=[file]).load_data()

                messages = [
                    ChatMessage(
                        role="user",
                        blocks=[
                            ImageBlock(path=file),
                            TextBlock(text="Extract the contents in the image. Do not add other thins."),
                        ],
                    )
                ]

                resp = self.llm.chat(messages)
                with open(f'{save_folder}/{image_name}.txt', 'w', encoding='utf-8') as f:
                    f.write(resp.message.content)
            
            mixed_files.append(f'{save_folder}/{file.split('/')[-1].split("\\")[-1]}')
            mixed_files.append(f'{save_folder}/{image_name}.txt')
        
        logger.info("Processed %d image files", len(file_path))

        documents = SimpleDirectoryReader(input_files=mixed_files).load_data()
        simple_parser = SimpleFileNodeParser(chunk_size=1000000, chunk_overlap=0)
        nodes = simple_parser.get_nodes_from_documents(documents)
        
        for temp_node in nodes:
            if isinstance(temp_node,ImageNode):
                img_path = temp_node.metadata['file_path']
                image_base64 = image_to_base64(img_path)
                temp_node.image = image_base64
        
        logger.debug("Generated %d image nodes", len(nodes))
        return nodes

    def create_auto_retriever(self, index):
        """Create auto retriever"""
        
        # Set vector info
        vector_store_info = VectorStoreInfo(
            content_info="Document content, including various formats such as text, images, etc",
            metadata_info=[
                MetadataInfo(
                    name="file_path",
                    type="str",
                    description="File path information, used to locate the original file"
                ),
                MetadataInfo(
                    name="file_type",
                    type="str",
                    description="File types such as txt, docx, pdf, image, etc"
                ),
                MetadataInfo(
                    name="page_number",
                    type="int",
                    description="Page numbers of PDF files, 0 for other file types"
                )
            ],
        )
        
        # Create automatic retriever
        retriever = VectorIndexAutoRetriever(
            index,
            vector_store_info=vector_store_info,
            similarity_top_k=3
        )
        
        return retriever
    
    def upload_documents(self, directory_path, file_paths, class_name="Documents_llama"):
        """upload files to Weaviate"""

        vector_store = self.setup_vector_store(class_name)
        
        ##################
        image_paths = []
        pdf_paths = []
        nonimage_paths = []
        for file in file_paths:
            if file.split('.')[-1] in ['jpg', 'jpeg', 'png']:
                image_paths.append(file)
            elif file.split('.')[-1] in ['pdf']:
                pdf_paths.append(file)
            else:
                nonimage_paths.append(file)
        # nonimages
        if len(nonimage_paths)>0:
            # Loading and processing documents
            nodes = self.load_and_process_documents2(directory_path)

            # 创建索引并存储文档
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
        # images
        if len(image_paths)>0:
            # Loading and processing documents
            nodes = self.load_and_process_images(directory_path, image_paths, class_name)
            # Create an index and store documents
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
        # pdf
        if len(pdf_paths)>0:
            # Loading and processing pdf
            nodes = self.load_and_process_pdf(pdf_paths)

            # Create an index and store documents
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
        ##################

        # # Loading and processing documents
        # nodes = self.load_and_process_documents(directory_path)
        
        # # Create an index and store documents
        # storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # index = VectorStoreIndex(
        #     nodes,
        #     storage_context=storage_context,
        #     embed_model=self.embed_model
        # )
        
        print(f"Uploaded {len(nodes)} nodes to Weaviate successfully")
        # return index
    
    def query_documents(self, query, class_name="Documents_llama", use_auto_retriever=True):
        """QUERY"""
        vector_store = WeaviateVectorStore(
            weaviate_client=self.client,
            index_name=class_name
        )
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        
        if use_auto_retriever:
            print("use_auto_retriever")
            print(self.llm)
            # Using automatic retriever
            retriever = self.create_auto_retriever(index)
            query_engine = RetrieverQueryEngine.from_args(
                retriever,
                node_postprocessors=[
                    MetadataReplacementPostProcessor(target_metadata_key="window")
                ],
                llm=self.llm
            )
        else:

            # Configure query engine
            query_engine = index.as_query_engine(
                vector_store_query_mode="hybrid",
                alpha=0.5,  # Mixed search weight
                similarity_top_k=3,
                node_postprocessors=[
                    MetadataReplacementPostProcessor(target_metadata_key="window")
                ],
                llm=self.llm
            )
        
        response = query_engine.query(query)
        return response
    
    def close(self):
        self.client.close()

# upload
def main_upload(collection_name, documents_dir, file_paths):
    # Initialize the automatic retriever
    retriever = WeaviateAutoRetriever(
        embedding_model_path="./models/bge-base-en-v1.5"
    )
    
    try:
        retriever.upload_documents(documents_dir, file_paths, collection_name)
        
        file_names = [f.split('/')[-1].split('\\')[-1] for f in file_paths]
        logger.info("Uploaded %d files to %s", len(file_names), collection_name)
        return f'Uploaded: {file_names} to {collection_name}'

    finally:
        retriever.close()

def main_query(collection_name, query):
    """Query documents from Weaviate collection"""
    retriever = WeaviateAutoRetriever(
        embedding_model_path="./models/bge-base-en-v1.5"
    )
    
    try:
        response = retriever.query_documents(query, collection_name, use_auto_retriever=True)
        
        logger.info("Query: %s", query)
        logger.debug("Response: %s", response)
        
        for node in response.source_nodes:
            logger.debug("Source: %s (similarity: %.3f)", 
                        node.metadata.get('file_path', 'unknown file'), 
                        node.score)
        
        return response    

    finally:
        retriever.close()


from weaviate.classes.query import BM25Operator

def search_bm25(collection_name, query_input):
    """Perform BM25 search on Weaviate collection"""
    try:
        client = weaviate.connect_to_local()
        collection = client.collections.use(collection_name)
        
        logger.debug("BM25 search query: %s", query_input)
        response = collection.query.bm25(
            query=query_input,
            operator=BM25Operator.or_(minimum_match=1),
            limit=3,
        )

        return response.objects
    finally:
        client.close()

def extract_file_filter(query):
    file_filter = []
    nonimg_matches = re.findall(r'\b[\w\-/\.]+\.(?:pdf|docx|md|json|html|htm|epub|pptx|csv|xml|eml|txt)\b', query, re.IGNORECASE)
    file_filter += nonimg_matches
    img_matches = re.findall(r'\b[\w\-/\.]+\.(?:jpg|jpeg|png|svg)\b', query, re.IGNORECASE)
    if img_matches:
        file_filter += img_matches
        for img_file in img_matches:
            file_filter.append(f"{img_file.split('.')[0]}.txt")
    return file_filter

def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        binary_data = f.read()
        base64_encoded = base64.b64encode(binary_data)
        return(base64_encoded.decode('utf-8'))

if __name__ == "__main__":
    query = "How is the hotline service acquired?"
    main_query("Documents_llama", query)
