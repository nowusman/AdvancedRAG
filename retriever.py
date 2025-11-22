import os
import json
import pymongo
import re
from dotenv import load_dotenv
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from llama_index.core import SimpleDirectoryReader, StorageContext, SummaryIndex, VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core.node_parser import LangchainNodeParser, SimpleFileNodeParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.docling import DoclingReader
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import load_index_from_storage
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core.tools import RetrieverTool
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import PydanticMultiSelector, PydanticSingleSelector
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
import base64

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
EMBED_MODEL_PATH = os.getenv('EMBED_MODEL_PATH')
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_PATH)
openai_mm_llm = OpenAIMultiModal(
    model="gpt-4o-mini", api_key=OPENAI_API_KEY, max_new_tokens=1500
)
RETRIEVER_MODEL = os.getenv('RETRIEVER_MODEL')
llm = OpenAI(model=RETRIEVER_MODEL)

class MongoDBChecker:
    def __init__(self, MONGO_URI, DB_NAME):
        self.MONGO_URI = MONGO_URI
        self.DB_NAME = DB_NAME

    def get_db_docs(self):
        mongo_client = pymongo.MongoClient(self.MONGO_URI)
        db = mongo_client[self.DB_NAME]
        collection_names = db.list_collection_names()
        data_collections = [c.split('/')[0] for c in collection_names if c.split('/')[1]=='data' and c.split('/')[0]!='indexes']
        if len(data_collections)==0:
            return(None)
        else:
            db_docs = {}
            for collection in data_collections:
                storage_context = StorageContext.from_defaults(
                    docstore=MongoDocumentStore.from_uri(uri=self.MONGO_URI, db_name=self.DB_NAME, namespace=collection),
                    index_store=MongoIndexStore.from_uri(uri=self.MONGO_URI, db_name=self.DB_NAME, namespace='indexes'),
                )
                all_ref_doc_info = storage_context.docstore.get_all_ref_doc_info()
                file_names = []
                if len(all_ref_doc_info)>0:
                    for key in all_ref_doc_info.keys():
                        file_names.append(all_ref_doc_info[key].metadata['file_name'])
                db_docs[collection] = file_names
            return(db_docs)

def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        binary_data = f.read()
        base64_encoded = base64.b64encode(binary_data)
        return(base64_encoded.decode('utf-8'))

class MongoDBCollectionManager:
    def __init__(self, MONGO_URI, DB_NAME, data_collection, PERSIST_DIR, IMAGE_FOLDER, INDEX_INFO_PATH):
        self.MONGO_URI = MONGO_URI
        self.DB_NAME = DB_NAME
        self.data_collection = data_collection
        self.PERSIST_DIR = PERSIST_DIR
        self.IMAGE_FOLDER = IMAGE_FOLDER
        self.INDEX_INFO_PATH = INDEX_INFO_PATH

        if not os.path.exists(os.path.join(self.PERSIST_DIR,'default__vector_store.json')):
            self.persist_exist = False
            self.storage_context = StorageContext.from_defaults(
                docstore=MongoDocumentStore.from_uri(uri=self.MONGO_URI, db_name=self.DB_NAME, namespace=self.data_collection),
                index_store=MongoIndexStore.from_uri(uri=self.MONGO_URI, db_name=self.DB_NAME, namespace='indexes'),
            )
        else:
            self.persist_exist = True
            self.storage_context = StorageContext.from_defaults(
                docstore=MongoDocumentStore.from_uri(uri=self.MONGO_URI, db_name=self.DB_NAME, namespace=self.data_collection),
                index_store=MongoIndexStore.from_uri(uri=self.MONGO_URI, db_name=self.DB_NAME, namespace='indexes'),
                persist_dir=self.PERSIST_DIR
            )

    def upload_files_from_filelist_nonimage(self, file_path, embed_model=embed_model):
        txt_paths = []
        docling_paths = []
        for file in file_path:
            if file.split('.')[-1]=='txt':
                txt_paths.append(file)
            else:
                docling_paths.append(file)
        # load txt
        if len(txt_paths)>0:
            txt_reader = SimpleDirectoryReader(input_files=txt_paths)
            txt_docs = txt_reader.load_data()
        else:
            txt_docs = []
        # docling reader
        if len(docling_paths)>0:
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
        else:
            docling_docs = []
        docs = txt_docs+docling_docs
        parser = LangchainNodeParser(RecursiveCharacterTextSplitter(separators='#'))
        nodes = parser.get_nodes_from_documents(docs)

        if len(nodes)==0:
            return('No nodes extracted!')
        else:
            self.storage_context.docstore.add_documents(nodes)
            print('Uploaded files to MongoDB!')
            ### check index existence
            try:
                with open(self.INDEX_INFO_PATH, 'r') as f:
                    index_info = json.load(f)
            except:
                index_info = {}

            if (f'{self.data_collection}_vector_index' in index_info.keys()):
                vector_id = index_info[f'{self.data_collection}_vector_index']
                vector_index = load_index_from_storage(
                    storage_context=self.storage_context, index_id=vector_id, embed_model=embed_model
                )
                vector_index.insert_nodes(nodes)
            else:
                vector_index = VectorStoreIndex(nodes, storage_context=self.storage_context, embed_model=embed_model)
                self.storage_context.persist()
                vector_id = vector_index.index_id
                index_info[f'{self.data_collection}_vector_index'] = vector_id
                if self.persist_exist==False:
                    self.storage_context = StorageContext.from_defaults(
                        docstore=MongoDocumentStore.from_uri(uri=self.MONGO_URI, db_name=self.DB_NAME, namespace=self.data_collection),
                        index_store=MongoIndexStore.from_uri(uri=self.MONGO_URI, db_name=self.DB_NAME, namespace='indexes'),
                        persist_dir=self.PERSIST_DIR
                    )
                    self.persist_exist=True

            with open(self.INDEX_INFO_PATH, 'w') as f:
                json.dump(index_info, f, indent=4)
            print('Created/updated indexes!')
    
    def upload_images_from_filelist(self, file_path, openai_mm_llm=llm, embed_model=embed_model):
        save_folder = os.path.join(self.IMAGE_FOLDER, self.data_collection)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        try:
            with open(self.INDEX_INFO_PATH, 'r') as f:
                index_info = json.load(f)
        except:
            index_info = {}

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

                resp = openai_mm_llm.chat(messages)
                with open(f'{save_folder}/{image_name}.txt', 'w', encoding='utf-8') as f:
                    f.write(resp.message.content)
                # image_desc = openai_mm_llm.complete(
                #     prompt="Extract the contents in the image. Do not add other thins.",
                #     # image_documents=image_doc,
                # )
                # with open(f'{save_folder}/{image_name}.txt', 'w', encoding='utf-8') as f:
                #     f.write(image_desc.text)
            
            mixed_files.append(f'{save_folder}/{file.split('/')[-1].split("\\")[-1]}')
            mixed_files.append(f'{save_folder}/{image_name}.txt')
        print('Image docs ready.')

        documents = SimpleDirectoryReader(input_files=mixed_files).load_data()
        simple_parser = SimpleFileNodeParser(chunk_size=1000000, chunk_overlap=0)
        nodes = simple_parser.get_nodes_from_documents(documents)
        # add base64 image data to ImageNode
        for temp_node in nodes:
            if isinstance(temp_node,ImageNode):
                img_path = temp_node.metadata['file_path']
                image_base64 = image_to_base64(img_path)
                temp_node.image = image_base64

        self.storage_context.docstore.add_documents(nodes)
        print('Uploaded images to MongoDB!')
        
        text_nodes = [n for n in nodes if not isinstance(n,ImageNode)]
        if (f'{self.data_collection}_vector_index' in index_info.keys()):
            vector_id = index_info[f'{self.data_collection}_vector_index']
            vector_index = load_index_from_storage(
                storage_context=self.storage_context, index_id=vector_id, embed_model=embed_model
            )
            vector_index.insert_nodes(text_nodes)
        else:
            vector_index = VectorStoreIndex(text_nodes, storage_context=self.storage_context, embed_model=embed_model)
            self.storage_context.persist()
            vector_id = vector_index.index_id
            index_info[f'{self.data_collection}_vector_index'] = vector_id
            if self.persist_exist==False:
                self.storage_context = StorageContext.from_defaults(
                    docstore=MongoDocumentStore.from_uri(uri=self.MONGO_URI, db_name=self.DB_NAME, namespace=self.data_collection),
                    index_store=MongoIndexStore.from_uri(uri=self.MONGO_URI, db_name=self.DB_NAME, namespace='indexes'),
                    persist_dir=self.PERSIST_DIR
                )
                self.persist_exist=True

        if (f'{self.data_collection}_multimodal_index' in index_info.keys()):
            multimodal_id = index_info[f'{self.data_collection}_multimodal_index']
            multimodal_index = load_index_from_storage(
                storage_context=self.storage_context, index_id=multimodal_id, embed_model=embed_model
            )
            multimodal_index.insert_nodes(nodes)
            self.storage_context.persist()
        else:
            multimodal_index = MultiModalVectorStoreIndex(
                nodes,
                storage_context=self.storage_context,
                embed_model = embed_model
            )
            self.storage_context.persist()
            multimodal_id = multimodal_index.index_id
            index_info[f'{self.data_collection}_multimodal_index'] = multimodal_id
               
        with open(self.INDEX_INFO_PATH, 'w') as f:
            json.dump(index_info, f, indent=4)
        print('Created/updated image indexes!')
            
    def upload_file(self, file_path, embed_model=embed_model, openai_mm_llm=openai_mm_llm):
        if type(file_path)==list:
            if len(file_path)==0:
                return('No files selected.')
            else:
                image_paths = []
                nonimage_paths = []
                for file in file_path:
                    if file.split('.')[-1] in ['jpg', 'jpeg', 'png', 'svg']:
                        image_paths.append(file)
                    else:
                        nonimage_paths.append(file)
                # nonimages
                if len(nonimage_paths)>0:
                    self.upload_files_from_filelist_nonimage(nonimage_paths, embed_model)
                # images
                if len(image_paths)>0:
                    self.upload_images_from_filelist(image_paths, llm, embed_model)
                return(f'Uploaded: {[f.split('/')[-1].split('\\')[-1] for f in file_path]} to {self.data_collection}')
        elif os.path.isdir(file_path):
            files = os.listdir(file_path)
            if len(files)==0:
                return('No files in the folder.')
            else:
                image_paths = []
                nonimage_paths = []
                for root, dirs, files in os.walk(file_path):
                    for file in files:
                        single_path = os.path.join(root, file)
                        if file.split('.')[-1] in ['jpg', 'png', 'svg']:
                            image_paths.append(single_path)
                        else:
                            nonimage_paths.append(single_path)
                # nonimages
                if len(nonimage_paths)>0:
                    self.upload_files_from_filelist_nonimage(nonimage_paths, embed_model)
                # images
                if len(image_paths)>0:
                    self.upload_images_from_filelist(image_paths, llm, embed_model)
                return(f'Uploaded: {[f.split('/')[-1].split('\\')[-1] for f in file_path]} to {self.data_collection}')
        elif os.path.isfile(file_path):
            # nonimages
            if file_path.split('.')[-1] not in ['jpg', 'png', 'svg']:
                self.upload_files_from_filelist_nonimage([file_path], embed_model)
            else:
                self.upload_images_from_filelist([file_path], llm, embed_model)
            return(f'Uploaded: {[f.split('/')[-1].split('\\')[-1] for f in file_path]} to {self.data_collection}')
    def delete_single_index(self, delete_index_id):
        with open(self.INDEX_INFO_PATH, 'r') as f:
            index_info = json.load(f)
        self.storage_context.index_store.delete_index_struct(delete_index_id)
        index_info_new = {k: v for k,v in index_info.items() if v!=delete_index_id}
        with open(self.INDEX_INFO_PATH, 'w') as f:
            json.dump(index_info_new, f, indent=4)

    def delete_files(self, file_names):
        all_ref_doc_info = self.storage_context.docstore.get_all_ref_doc_info()
        if len(all_ref_doc_info)==0:
            return(f'No files in collection!')
        else:
            for filename in file_names:
                if filename.split('.')[-1] in ['jpg', 'png', 'svg']:
                    fns = [filename, f'{filename.split('.')[0]}.txt']
                else:
                    fns = [filename]
                for fn in fns:
                    ref_doc_ids = [id for id in all_ref_doc_info.keys() if all_ref_doc_info[id].metadata['file_name']==fn]
                    if len(ref_doc_ids)==0:
                        return(f'[File Check Error]: {fn} not found.')
                    else:
                        node_ids = []
                        for id in all_ref_doc_info.keys():
                            if all_ref_doc_info[id].metadata['file_name']==fn:
                                node_ids += all_ref_doc_info[id].node_ids
                        try:
                            with open(self.INDEX_INFO_PATH, 'r') as f:
                                index_info = json.load(f)
                            for ref_id in ref_doc_ids:
                                self.storage_context.docstore.delete_ref_doc(ref_id)
                            print(f'{fn} deleted from Docstore!')
                            for ref_id in ref_doc_ids:
                                self.storage_context.vector_store.delete(ref_id)
                            self.storage_context.persist()
                            print(f'{fn} deleted from Vector Store!')
                            index_structs = self.storage_context.index_store.index_structs()
                            for index_struct in index_structs:
                                index_id = index_struct.index_id
                                index = load_index_from_storage(
                                        self.storage_context,
                                        index_id=index_id
                                    )
                                index_type = index_struct.__class__.__name__

                                if (index_type == "IndexDict") or (index_type == "MultiModelIndexDict"):
                                    for node_id in node_ids:
                                        if node_id in index.index_struct.nodes_dict.keys():
                                            del index.index_struct.nodes_dict[node_id]
                                    if len(index.index_struct.nodes_dict.keys())>0:
                                        index.storage_context.index_store.add_index_struct(index.index_struct)
                                    else:
                                        self.delete_single_index(index_id)

                            print(f'Nodes of {fn} deleted from Index Store!')
                        except Exception as e:
                            return(f'[File Deletion Error]: {e}')
            return(f'Deleted: {fns} from {self.data_collection}')

    def delete_collection(self):
        all_ref_doc_info = self.storage_context.docstore.get_all_ref_doc_info()
        if len(all_ref_doc_info)==0:
            return(f'No files in collection!')
        else:
            node_ids = []
            for id in all_ref_doc_info.keys():
                node_ids += all_ref_doc_info[id].node_ids
            try:
                # delete docstore
                mongo_client = pymongo.MongoClient(self.MONGO_URI)
                db = mongo_client[self.DB_NAME]
                db.drop_collection(f"{self.data_collection}/data")
                db.drop_collection(f"{self.data_collection}/metadata")
                db.drop_collection(f"{self.data_collection}/ref_doc_info")
                print(f'Deleted {self.data_collection} docstore.')
                # delete vectorstore
                for ref_id in all_ref_doc_info.keys():
                    self.storage_context.vector_store.delete(ref_id)
                self.storage_context.persist()
                print(f'Deleted relevant info in {self.data_collection} vectorstore.')
                # delete indexes
                with open(self.INDEX_INFO_PATH, 'r') as f:
                    index_info = json.load(f)

                if f'{self.data_collection}_vector_index' in index_info.keys():
                    index_id = index_info[f'{self.data_collection}_vector_index']
                    self.delete_single_index(index_id)
                if f'{self.data_collection}_keyword_index' in index_info.keys():
                    index_id = index_info[f'{self.data_collection}_keyword_index']
                    self.delete_single_index(index_id)
                if f'{self.data_collection}_multimodal_index' in index_info.keys():
                    index_id = index_info[f'{self.data_collection}_multimodal_index']
                    self.delete_single_index(index_id)    
                print(f'Deleted nodes in {self.data_collection} indexstore.')
            except Exception as e:
                return(f'[Collection Deletion Error]: {e}')
        return(f'Deleted {self.data_collection}.')

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

class IntelligentRetriever:
    def __init__(self, MONGO_URI, DB_NAME, COLLECTION_NAME, PERSIST_DIR, INDEX_INFO_PATH):
        self.MONGO_URI = MONGO_URI
        self.DB_NAME = DB_NAME
        self.COLLECTION_NAME = COLLECTION_NAME
        self.PERSIST_DIR = PERSIST_DIR
        self.INDEX_INFO_PATH = INDEX_INFO_PATH
        self.storage_context = StorageContext.from_defaults(
            docstore=MongoDocumentStore.from_uri(uri=self.MONGO_URI, db_name=self.DB_NAME, namespace=self.COLLECTION_NAME),
            index_store=MongoIndexStore.from_uri(uri=self.MONGO_URI, db_name=self.DB_NAME, namespace='indexes'),
            persist_dir=self.PERSIST_DIR
        )
        self.retriever = None

    def build_retriever(self, file_filter, embed_model=embed_model, llm=llm):
        try:
            with open(self.INDEX_INFO_PATH, 'r') as f:
                index_info = json.load(f)
        except:
            return('[Index Loading Error]: There is no existing index. Please upload files and create first.')
        if len(index_info)==0:
            return('[Index Loading Error]: There is no existing index. Please upload files and create first.')
        else:
            tools = []
            if file_filter:
                filters = MetadataFilters(filters=[
                    MetadataFilter(key="file_name", operator="in", value=file_filter)
                ])
            
            if f'{self.COLLECTION_NAME}_vector_index' in index_info.keys():
                vector_id = index_info[f'{self.COLLECTION_NAME}_vector_index']
                vector_index = load_index_from_storage(
                    storage_context=self.storage_context, index_id=vector_id, embed_model=embed_model
                )
                if file_filter:
                    vector_retriever = vector_index.as_retriever(similarity_top_k=3, filters=filters)
                    all_docs = list(self.storage_context.docstore.docs.values())
                    filtered_docs = []
                    for file_name in file_filter:
                        filtered_docs += [
                            doc for doc in all_docs
                            if doc.metadata.get("file_name") == file_name
                        ]
                    top_k = min(3, len(filtered_docs))
                    bm25_retriever = BM25Retriever.from_defaults(
                        nodes = filtered_docs,
                        similarity_top_k=top_k,
                    )
                else:
                    vector_retriever = vector_index.as_retriever(similarity_top_k=3)
                    all_docs = list(self.storage_context.docstore.docs.values())
                    top_k = min(3, len(all_docs))
                    bm25_retriever = BM25Retriever.from_defaults(
                        docstore = self.storage_context.docstore,
                        similarity_top_k=top_k
                    )
                text_retriever = QueryFusionRetriever(
                    retrievers=[
                        vector_retriever,
                        bm25_retriever
                    ],
                    retriever_weights=[
                        0.4, # vector retriever weight
                        0.6 # bm25 retriever weight
                    ],
                    num_queries=1, 
                    similarity_top_k=3,
                    mode='dist_based_score',
                    use_async=False
                )
                text_tool = RetrieverTool.from_defaults(
                    retriever=text_retriever,
                    description=(
                        f"Useful for retrieving specific text context."
                    ),
                )
                tools.append(text_tool)


            if f'{self.COLLECTION_NAME}_multimodal_index' in index_info.keys():
                multimodal_id = index_info[f'{self.COLLECTION_NAME}_multimodal_index']
                multimodal_index = load_index_from_storage(
                    storage_context=self.storage_context, index_id=multimodal_id, embed_model=embed_model
                )
                if file_filter:
                    multimodal_retriever = multimodal_index.as_retriever(
                        similarity_top_k=3, image_similarity_top_k=3, filters=filters
                    )
                else:
                    multimodal_retriever = multimodal_index.as_retriever(
                        similarity_top_k=3, image_similarity_top_k=3
                    )
                multimodal_tool = RetrieverTool.from_defaults(
                    retriever=multimodal_retriever,
                    description=(
                        f"Useful for retrieving both texts and images, especially when asked for specific image(s)."
                    ),
                )
                tools.append(multimodal_tool)

            retriever = RouterRetriever(
                # selector=PydanticMultiSelector.from_defaults(llm=llm),
                selector=PydanticSingleSelector.from_defaults(llm=llm),
                retriever_tools=tools
            )
            self.retriever = retriever
            ########################################3
            print(retriever)
            print(llm)
            ########################################3

            if self.retriever==None:
                return('Retriever cannot be built.')
            else:
                return('Retriever created.')

    def nodes_retrieval(self, query):
        if self.retriever!=None:
            nodes = self.retriever.retrieve(query)
            if len(nodes)>0:
                return(nodes)
            else:
                return('No related information. Please check whether relevant data collection is empty.')
        else:
            return('Retriever can not built! Please check the data collection.')
    
    def plot_images(self, image_paths):
        images_shown = 0
        plt.figure(figsize=(16, 9))
        for img_path in image_paths:
            if os.path.isfile(img_path):
                image = Image.open(img_path)

                plt.subplot(2, 3, images_shown + 1)
                plt.imshow(image)
                plt.xticks([])
                plt.yticks([])

                images_shown += 1
                if images_shown >= 9:
                    break

    def display_nodes(self, nodes):
        retrieved_image = []
        for res_node in nodes:
            if isinstance(res_node.node, ImageNode):
                retrieved_image.append(res_node.node.metadata["file_path"])
            else:
                display_source_node(res_node, source_length=200)
        if len(retrieved_image)>0:
            self.plot_images(retrieved_image)

