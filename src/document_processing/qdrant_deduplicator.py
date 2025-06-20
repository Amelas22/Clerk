"""
Document deduplication using Qdrant as the registry.
"""

import hashlib
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, PayloadSchemaType
)

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """Record of a document in the registry"""
    id: str
    document_hash: str
    file_name: str
    file_path: str
    case_name: str
    first_seen_at: datetime
    last_duplicate_found: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    duplicate_locations: List[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "document_hash": self.document_hash,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "case_name": self.case_name,
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_duplicate_found": self.last_duplicate_found.isoformat() if self.last_duplicate_found else None,
            "metadata": self.metadata or {},
            "duplicate_locations": self.duplicate_locations or []
        }


class QdrantDocumentDeduplicator:
    """
    Document deduplication system using Qdrant.
    Uses a separate collection to track document hashes and prevent duplicates.
    """
    
    def __init__(self):
        """Initialize with Qdrant client"""
        self.client = QdrantClient(
            url=settings.qdrant.url,
            api_key=settings.qdrant.api_key,
            prefer_grpc=settings.qdrant.prefer_grpc,
            timeout=settings.qdrant.timeout
        )
        
        self.collection_name = settings.qdrant.registry_collection_name
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Ensure document registry collection exists"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self._create_registry_collection()
                logger.info(f"Created document registry collection: {self.collection_name}")
            else:
                logger.info(f"Document registry collection exists: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring registry collection exists: {str(e)}")
            raise
    
    def _create_registry_collection(self):
        """Create the document registry collection"""
        # Use a small dummy vector since Qdrant requires vectors
        # We'll use 8 dimensions and store the hash as the vector
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=8,  # Small vector just for storage
                distance=Distance.EUCLID
            ),
            on_disk_payload=False  # Keep in RAM for fast lookups
        )
        
        # Create payload indexes for efficient filtering
        indexes = [
            ("document_hash", PayloadSchemaType.KEYWORD),
            ("case_name", PayloadSchemaType.KEYWORD),
            ("file_name", PayloadSchemaType.KEYWORD),
            ("first_seen_at", PayloadSchemaType.DATETIME),
        ]
        
        for field_name, field_type in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.debug(f"Created index for {field_name}")
            except Exception as e:
                logger.debug(f"Index {field_name} might already exist: {str(e)}")
    
    def calculate_document_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of document content
        
        Args:
            content: Document content as bytes
            
        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(content).hexdigest()
    
    def _hash_to_vector(self, doc_hash: str) -> List[float]:
        """Convert hash to a small vector for Qdrant storage
        
        Args:
            doc_hash: Document hash string
            
        Returns:
            8-dimensional vector
        """
        # Take first 8 bytes of hash and convert to floats
        hash_bytes = bytes.fromhex(doc_hash[:16])
        return [float(b) / 255.0 for b in hash_bytes]
    
    def check_document_exists(self, doc_hash: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if document with given hash already exists
        
        Args:
            doc_hash: Document hash to check
            
        Returns:
            Tuple of (exists, existing_record)
        """
        try:
            # Search by document hash
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_hash",
                            match=MatchValue(value=doc_hash)
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            if results[0]:
                record = results[0][0].payload
                return True, record
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking document existence: {str(e)}")
            return False, None
    
    def register_new_document(self, doc_hash: str, file_name: str, 
                            file_path: str, case_name: str,
                            metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Register a new document in the deduplication system
        
        Args:
            doc_hash: Document hash
            file_name: Name of the file
            file_path: Path to file in Box
            case_name: Case this document belongs to
            metadata: Additional metadata
            
        Returns:
            Document record
        """
        try:
            doc_record = DocumentRecord(
                id=str(uuid.uuid4()),
                document_hash=doc_hash,
                file_name=file_name,
                file_path=file_path,
                case_name=case_name,
                first_seen_at=datetime.utcnow(),
                metadata=metadata or {},
                duplicate_locations=[]
            )
            
            # Create point for Qdrant
            point = PointStruct(
                id=doc_record.id,
                vector=self._hash_to_vector(doc_hash),
                payload=doc_record.to_dict()
            )
            
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True
            )
            
            logger.info(f"Registered new document: {file_name} (hash: {doc_hash[:8]}...)")
            return doc_record.to_dict()
            
        except Exception as e:
            logger.error(f"Error registering document: {str(e)}")
            raise
    
    def add_duplicate_location(self, doc_hash: str, file_path: str, case_name: str):
        """Record a duplicate file location
        
        Args:
            doc_hash: Document hash
            file_path: Path where duplicate was found
            case_name: Case where duplicate was found
        """
        try:
            # Get existing record
            exists, record = self.check_document_exists(doc_hash)
            
            if not exists:
                logger.warning(f"Cannot add duplicate location - document not found: {doc_hash[:8]}...")
                return
            
            # Update duplicate locations
            duplicate_locations = record.get("duplicate_locations", [])
            duplicate_locations.append({
                "file_path": file_path,
                "case_name": case_name,
                "found_at": datetime.utcnow().isoformat()
            })
            
            # Update the record
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={
                    "duplicate_locations": duplicate_locations,
                    "last_duplicate_found": datetime.utcnow().isoformat()
                },
                points=[record["id"]]
            )
            
            logger.info(f"Added duplicate location for {doc_hash[:8]}... in case {case_name}")
            
        except Exception as e:
            logger.error(f"Error adding duplicate location: {str(e)}")
    
    def get_document_by_hash(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        """Get document record by hash
        
        Args:
            doc_hash: Document hash
            
        Returns:
            Document record or None
        """
        exists, record = self.check_document_exists(doc_hash)
        return record if exists else None
    
    def get_case_documents(self, case_name: str) -> List[Dict[str, Any]]:
        """Get all unique documents for a case
        
        Args:
            case_name: Case name
            
        Returns:
            List of document records
        """
        try:
            # Use scroll with proper offset handling
            documents = []
            offset = None
            
            while True:
                # The scroll method returns a tuple of (batch, offset)
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="case_name",
                                match=MatchValue(value=case_name)
                            )
                        ]
                    ),
                    limit=100,  # Process in smaller batches
                    offset=offset,  # Pass the offset for pagination
                    with_payload=True,
                    with_vectors=False
                )
                
                # Check if we got a valid response
                if not scroll_result or len(scroll_result) < 2:
                    logger.warning(f"Unexpected scroll response format for case {case_name}")
                    break
                
                batch, new_offset = scroll_result
                
                # Add points from this batch to our result
                if batch:
                    for point in batch:
                        if hasattr(point, 'payload') and point.payload:
                            documents.append(point.payload)
                
                # If no new offset or empty batch, we're done
                if new_offset is None or not batch:
                    break
                
                # Update offset for next iteration
                offset = new_offset
            
            logger.debug(f"Retrieved {len(documents)} documents for case {case_name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error getting case documents: {str(e)}")
            return []
    
    def get_duplicates_for_document(self, doc_hash: str) -> List[Dict[str, str]]:
        """Get all duplicate locations for a document
        
        Args:
            doc_hash: Document hash
            
        Returns:
            List of duplicate locations
        """
        record = self.get_document_by_hash(doc_hash)
        if record:
            return record.get("duplicate_locations", [])
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get deduplication statistics
        
        Returns:
            Dictionary with statistics
        """
        try:
            # Get collection info
            info = self.client.get_collection(self.collection_name)
            total_documents = info.points_count
            
            # Get all documents to calculate statistics
            all_docs = []
            offset = None
            
            while True:
                batch, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                all_docs.extend(batch)
                
                if offset is None:
                    break
            
            # Calculate statistics
            total_duplicates = 0
            cases = set()
            
            for doc in all_docs:
                payload = doc.payload
                cases.add(payload.get("case_name"))
                duplicates = payload.get("duplicate_locations", [])
                total_duplicates += len(duplicates)
            
            return {
                "total_unique_documents": total_documents,
                "total_duplicate_instances": total_duplicates,
                "total_cases": len(cases),
                "average_duplicates_per_document": total_duplicates / total_documents if total_documents > 0 else 0,
                "cases": list(cases)
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {
                "total_unique_documents": 0,
                "total_duplicate_instances": 0,
                "total_cases": 0,
                "average_duplicates_per_document": 0,
                "error": str(e)
            }
    
    def cleanup_case(self, case_name: str) -> int:
        """Remove all documents for a specific case
        
        Args:
            case_name: Case to clean up
            
        Returns:
            Number of documents removed
        """
        try:
            # Get all documents for the case
            case_docs = self.get_case_documents(case_name)
            
            if not case_docs:
                return 0
            
            # Delete by IDs
            doc_ids = [doc["id"] for doc in case_docs]
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=doc_ids
            )
            
            logger.info(f"Cleaned up {len(doc_ids)} documents for case {case_name}")
            return len(doc_ids)
            
        except Exception as e:
            logger.error(f"Error cleaning up case: {str(e)}")
            return 0
    
    def export_registry(self) -> Dict[str, List[Dict[str, Any]]]:
        """Export the entire document registry
        
        Returns:
            Dictionary with all documents by case
        """
        try:
            stats = self.get_statistics()
            registry = {}
            
            for case in stats.get("cases", []):
                registry[case] = self.get_case_documents(case)
            
            return registry
            
        except Exception as e:
            logger.error(f"Error exporting registry: {str(e)}")
            return {}


# Backward compatibility alias
DocumentDeduplicator = QdrantDocumentDeduplicator