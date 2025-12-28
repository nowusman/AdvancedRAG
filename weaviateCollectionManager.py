import os
import weaviate
from weaviate import EmbeddedOptions
import weaviate.classes.config as wvcc
import weaviate.collections.classes.filters as filters
from dotenv import load_dotenv
from typing import List, Dict, Any
import pandas as pd
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class WeaviateCollectionManager:
    def __init__(self):
        # connect to Weaviate
        self.client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051
        )
    
    def get_uploaded_files(self, class_name: str = "Documents_llama") -> List[Dict[str, Any]]:
        """
        Get uploaded file list
        
        Args:
            class_name: Weaviate collection name
            
        Returns:
            dictionary include doc's info
        """
        try:
            # get collection
            collection = self.client.collections.get(class_name)
            
            # query all objects, return meta data
            response = collection.query.fetch_objects(
                include_vector=False,  # donot return vector data
                return_properties=["file_path", "file_type", "content", "page_number"],
                limit=1000  # 
            )
            
            # 
            files_info = []
            seen_files = set()  # Used for deduplication
            
            for obj in response.objects:
                file_path = obj.properties.get("file_path", "")
                
                # De duplication, keep only the unique file path
                if file_path and file_path not in seen_files:
                    seen_files.add(file_path)
                    files_info.append({
                        "file_path": file_path,
                        "file_name": file_path.split('/')[-1],
                        "file_type": obj.properties.get("file_type", ""),
                        "content": obj.properties.get("content", ""),
                        "chunk_count": 1  # Initialize to 1,count later
                    })
                elif file_path in seen_files:
                    # 
                    for file_info in files_info:
                        if file_info["file_path"] == file_path:
                            file_info["chunk_count"] += 1
                            break
            
            return files_info
            
        except Exception as e:
            logger.error("Error retrieving uploaded files from %s: %s", class_name, str(e))
            return []
    
    def get_file_chunks(self, class_name: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Retrieve all text blocks of the specified file
        
        Args:
            class_name: Weaviate collection name
            file_path: file path
            
        Returns:
            dictionary include doc's info
        """
        try:
            # 
            collection = self.client.collections.get(class_name)
            
            # query all objects, return meta data
            response = collection.query.fetch_objects(
                filters=filters.Filter.by_property("file_path").equal(file_path),
                include_vector=False,
                return_properties=["text", "file_type", "content"],
                limit=1000  # 
            )
            
            # 
            chunks_info = []
            for obj in response.objects:
                chunks_info.append({
                    "text": obj.properties.get("text", "")[:200] + "..." if len(obj.properties.get("text", "")) > 200 else obj.properties.get("text", ""),
                    "file_type": obj.properties.get("file_type", ""),
                    "content": obj.properties.get("content", ""),
                    "object_id": obj.uuid
                })
            
            return chunks_info
            
        except Exception as e:
            logger.error("Error obtaining file chunks for %s: %s", file_path, str(e))
            return []
    
    def get_collection_stats(self, class_name: str = "Documents_llama") -> Dict[str, Any]:
        """
        Obtain statistical information of the collection
        
        Args:
            class_name: Weaviate collection
            
        Returns:
            Dictionary containing statistical information
        """
        try:
            # 获
            collection = self.client.collections.get(class_name)
            
            # obj count
            total_objects = collection.aggregate.over_all(total_count=True).total_count
            
            # Retrieve file type statistics
            file_type_stats = {}
            response = collection.aggregate.group_by(
                ["file_type"],
                filters=filters.Filter.by_property("file_type").is_not_null(),
                object_limit=1000,
                total_count=True
            )
            
            for group in response.groups:
                file_type = group.grouped_by.value if hasattr(group.grouped_by, 'value') else "unknown"
                file_type_stats[file_type] = group.total_count
            
            # Retrieve the number of files (based on file_cath deduplication)
            files_info = self.get_uploaded_files(class_name)
            total_files = len(files_info)
            
            return {
                "total_objects": total_objects,
                "total_files": total_files,
                "file_type_stats": file_type_stats
            }
            
        except Exception as e:
            logger.error("Error obtaining collection stats for %s: %s", class_name, str(e))
            return {}
    
    def display_uploaded_files(self, class_name: str = "Documents_llama"):
        """
        Display the list of uploaded files
        
        Args:
            class_name: Weaviate collection
        
        Returns:
            dict: Dictionary mapping collection name to files info
        """
        files_info = self.get_uploaded_files(class_name)
        
        if not files_info:
            logger.warning("No files in collection '%s' or collection does not exist", class_name)
            return {}
        
        df = pd.DataFrame(files_info)
        logger.info("Files in collection '%s':\n%s", class_name, df.to_string(index=False))

        return {class_name: files_info}
    
    def close(self):
        """close connection"""
        self.client.close()

    def delete_collection(self, class_name):
        """Delete a Weaviate collection"""
        if self.client.collections.exists(class_name):
            self.client.collections.delete(class_name)
            logger.info("Collection '%s' deleted successfully", class_name)
            return f'Collection {class_name} has been deleted!'
        else:
            logger.warning("Collection '%s' does not exist", class_name)
            return f'Collection {class_name} does not exist!'

    def delete_file_objects(self, class_name: str, file_path: str) -> int:
        """
        Args:
            class_name: collection name
            file_path: delete file path
            
        Returns:
            deleted file object count
        """
        try:
            if not self.client.collections.exists(class_name):
                logger.warning("Collection '%s' does not exist", class_name)
                return 0
            
            collection = self.client.collections.get(class_name)
            
            response = collection.query.fetch_objects(
                filters=filters.Filter.by_property("file_path").equal(file_path),
                include_vector=False,
                return_properties=[],  
                limit=1000  
            )
            
            object_ids = [obj.uuid for obj in response.objects]
            
            if not object_ids:
                logger.warning("File '%s' not found in collection '%s'", file_path, class_name)
                return 0
            
            deleted_count = 0
            for obj_id in object_ids:
                try:
                    collection.data.delete_by_id(obj_id)
                    deleted_count += 1
                except Exception as e:
                    logger.error("Failed to delete object %s: %s", obj_id, str(e))
            
            logger.info("Deleted %d objects for file '%s'", deleted_count, file_path)
            return deleted_count
            
        except Exception as e:
            logger.error("Error deleting file '%s': %s", file_path, str(e))
            raise

def delete_collection(class_name):
    manager = WeaviateCollectionManager()
    try:
        msg = manager.delete_collection(class_name)
        return msg
    finally:
        manager.close()

def delete_file(class_name, file_path):
    manager = WeaviateCollectionManager()
    try:
        msg = manager.delete_file_objects(class_name, file_path)
        return msg
    finally:
        manager.close()

def display_uploaded_files(class_name):
    # initial
    manager = WeaviateCollectionManager()
    
    try:
        # uploaded files
        files_info = manager.display_uploaded_files(class_name)
        
        return files_info
            
    finally:
        manager.close()

if __name__ == "__main__":
    display_uploaded_files("Documents_llama")
