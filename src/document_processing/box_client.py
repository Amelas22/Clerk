"""
Box API client for accessing and downloading case documents.
Handles authentication, folder traversal, and file downloads.
"""

import io
import logging
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass
from datetime import datetime

from boxsdk import Client, JWTAuth
from boxsdk.object.folder import Folder
from boxsdk.object.file import File

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class BoxDocument:
    """Represents a document from Box"""
    file_id: str
    name: str
    path: str
    case_name: str
    size: int
    modified_at: datetime
    parent_folder_id: str
    folder_path: List[str]

class BoxClient:
    """Manages Box API connections and file operations"""
    
    def __init__(self):
        """Initialize Box client with JWT authentication"""
        self.client = self._create_client()
        
    def _create_client(self) -> Client:
        """Create authenticated Box client"""
        try:
            auth = JWTAuth(
                client_id=settings.box.client_id,
                client_secret=settings.box.client_secret,
                enterprise_id=settings.box.enterprise_id,
                jwt_key_id=settings.box.jwt_key_id,
                rsa_private_key_data=settings.box.private_key,
                rsa_private_key_passphrase=settings.box.passphrase,
            )
            
            auth.authenticate_instance()
            client = Client(auth)
            
            # Test connection
            current_user = client.user().get()
            logger.info(f"Successfully connected to Box as {current_user.name}")
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to create Box client: {str(e)}")
            raise
    
    def get_folder_path(self, folder: Folder) -> Tuple[str, List[str]]:
        """Get the full path of a folder
        
        Returns:
            Tuple of (case_name, folder_path_list)
        """
        path_parts = []
        current = folder
        case_name = folder.name  # Default to current folder name
        
        while current.id != "0":  # "0" is the root folder
            path_parts.insert(0, current.name)
            try:
                parent = current.get_parent()
                if parent and parent.id != "0":
                    current = parent
                    case_name = parent.name  # Keep updating to get top-level
                else:
                    break
            except:
                break
                
        return case_name, path_parts
    
    def traverse_folder(self, parent_folder_id: str) -> Generator[BoxDocument, None, None]:
        """Recursively traverse folder and yield all PDF documents
        
        Args:
            parent_folder_id: Box folder ID to start traversal
            
        Yields:
            BoxDocument objects for each PDF found
        """
        try:
            folder = self.client.folder(folder_id=parent_folder_id).get()
            case_name, folder_path = self.get_folder_path(folder)
            
            logger.info(f"Traversing folder: {'/'.join(folder_path)} (Case: {case_name})")
            
            # Get all items in folder
            items = folder.get_items()
            
            for item in items:
                if item.type == "file":
                    # Check if it's a PDF
                    if item.name.lower().endswith('.pdf'):
                        file_info = self.client.file(file_id=item.id).get()
                        
                        # Create BoxDocument
                        doc = BoxDocument(
                            file_id=item.id,
                            name=item.name,
                            path=f"{'/'.join(folder_path)}/{item.name}",
                            case_name=case_name,
                            size=file_info.size,
                            modified_at=datetime.fromisoformat(
                                file_info.modified_at.replace('Z', '+00:00')
                            ),
                            parent_folder_id=parent_folder_id,
                            folder_path=folder_path
                        )
                        
                        yield doc
                        
                elif item.type == "folder":
                    # Recursively process subfolders
                    yield from self.traverse_folder(item.id)
                    
        except Exception as e:
            logger.error(f"Error traversing folder {parent_folder_id}: {str(e)}")
            raise
    
    def download_file(self, file_id: str) -> bytes:
        """Download file content from Box
        
        Args:
            file_id: Box file ID
            
        Returns:
            File content as bytes
        """
        try:
            file = self.client.file(file_id=file_id)
            content = io.BytesIO()
            file.download_to(content)
            content.seek(0)
            return content.read()
            
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {str(e)}")
            raise
    
    def get_file_info(self, file_id: str) -> Dict:
        """Get detailed file information
        
        Args:
            file_id: Box file ID
            
        Returns:
            Dictionary with file metadata
        """
        try:
            file = self.client.file(file_id=file_id).get()
            
            return {
                "id": file.id,
                "name": file.name,
                "size": file.size,
                "created_at": file.created_at,
                "modified_at": file.modified_at,
                "sha1": file.sha1,
                "parent": {
                    "id": file.parent.id,
                    "name": file.parent.name
                } if file.parent else None
            }
            
        except Exception as e:
            logger.error(f"Error getting file info for {file_id}: {str(e)}")
            raise
    
    def check_connection(self) -> bool:
        """Test Box API connection
        
        Returns:
            True if connection is working
        """
        try:
            self.client.user().get()
            return True
        except:
            return False