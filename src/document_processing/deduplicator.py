"""
Document deduplication module.
Handles hash-based duplicate detection and tracking.
"""

import hashlib
import logging
import json
import asyncio
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from urllib.parse import urljoin

from supabase import create_client, Client, acreate_client, AsyncClient
from postgrest.exceptions import APIError

from config.settings import settings

logger = logging.getLogger(__name__)

class DocumentDeduplicator:
    """Manages document deduplication using hash-based detection"""
    
    def __init__(self):
        # Use the base Supabase URL, not the REST endpoint URL
        supabase_url = settings.database.supabase_url
        supabase_key = settings.database.supabase_service_key
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure the document tracking table exists"""
        # Note: In production, this should be done via migrations
        # This is a simplified version for the initial implementation
        try:
            # Try to query the table
            self.supabase.table("document_registry").select("id").limit(1).execute()
            logger.info("Document registry table exists")
        except APIError as e:
            logger.warning(f"Document registry table might not exist: {str(e)}")
            # In production, handle table creation properly
    
    def calculate_document_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of document content
        
        Args:
            content: Document content as bytes
            
        Returns:
            Hex string of document hash
        """
        return hashlib.sha256(content).hexdigest()
    
    def check_document_exists(self, document_hash: str) -> Tuple[bool, Optional[Dict]]:
        """Check if document with given hash already exists
        
        Args:
            document_hash: SHA-256 hash of document
            
        Returns:
            Tuple of (exists: bool, existing_record: dict or None)
        """
        try:
            result = self.supabase.table("document_registry").select("*").eq(
                "document_hash", document_hash
            ).execute()
            
            if result.data and len(result.data) > 0:
                return True, result.data[0]
            
            return False, None
            
        except APIError as e:
            logger.error(f"Error checking document existence: {str(e)}")
            raise
    
    async def check_document_exists_async(self, document_hash: str) -> Tuple[bool, Optional[Dict]]:
        """Async version of check_document_exists
        
        Args:
            document_hash: SHA-256 hash of document
            
        Returns:
            Tuple of (exists: bool, existing_record: dict or None)
        """
        try:
            # Create a new async client for this operation
            supabase_url = settings.database.supabase_url
            supabase_key = settings.database.supabase_service_key
            async_client: AsyncClient = await acreate_client(supabase_url, supabase_key)
            
            result = await async_client.table("document_registry").select("*").eq(
                "document_hash", document_hash
            ).execute()
            
            # Close the async client connection
            await async_client.aclose()
            
            if result.data and len(result.data) > 0:
                return True, result.data[0]
            
            return False, None
            
        except APIError as e:
            logger.error(f"Error checking document existence (async): {str(e)}")
            raise
    
    def register_new_document(self, document_hash: str, file_name: str, 
                            file_path: str, case_name: str, 
                            metadata: Dict = None) -> Dict:
        """Register a new document in the deduplication system
        
        Args:
            document_hash: SHA-256 hash of document
            file_name: Name of the file
            file_path: Full path in Box
            case_name: Name of the case this document belongs to
            metadata: Additional metadata to store
            
        Returns:
            Created document record
        """
        try:
            record = {
                "document_hash": document_hash,
                "file_name": file_name,
                "file_path": file_path,
                "case_name": case_name,
                "first_seen_at": datetime.utcnow().isoformat(),
                "metadata": metadata or {},
                "duplicate_locations": []
            }
            
            result = self.supabase.table("document_registry").insert(record).execute()
            
            logger.info(f"Registered new document: {file_name} (hash: {document_hash[:8]}...)")
            return result.data[0]
            
        except APIError as e:
            logger.error(f"Error registering document: {str(e)}")
            raise
    
    def add_duplicate_location(self, document_hash: str, duplicate_path: str, 
                             case_name: str) -> Dict:
        """Add a duplicate location for an existing document
        
        Args:
            document_hash: SHA-256 hash of document
            duplicate_path: Path where duplicate was found
            case_name: Case where duplicate was found
            
        Returns:
            Updated document record
        """
        try:
            # Get existing record
            exists, record = self.check_document_exists(document_hash)
            
            if not exists:
                raise ValueError(f"Document with hash {document_hash} not found")
            
            # Update duplicate locations
            duplicate_info = {
                "path": duplicate_path,
                "case_name": case_name,
                "found_at": datetime.utcnow().isoformat()
            }
            
            duplicate_locations = record.get("duplicate_locations", [])
            
            # Check if this duplicate is already recorded
            existing = any(
                dup["path"] == duplicate_path 
                for dup in duplicate_locations
            )
            
            if not existing:
                duplicate_locations.append(duplicate_info)
                
                # Update record
                result = self.supabase.table("document_registry").update({
                    "duplicate_locations": duplicate_locations,
                    "last_duplicate_found": datetime.utcnow().isoformat()
                }).eq("document_hash", document_hash).execute()
                
                logger.info(
                    f"Added duplicate location for document {record['file_name']}: "
                    f"{duplicate_path}"
                )
                
                return result.data[0]
            else:
                logger.info(f"Duplicate location already recorded: {duplicate_path}")
                return record
                
        except APIError as e:
            logger.error(f"Error adding duplicate location: {str(e)}")
            raise
    
    def get_document_info(self, document_hash: str) -> Optional[Dict]:
        """Get full information about a document by hash
        
        Args:
            document_hash: SHA-256 hash of document
            
        Returns:
            Document record or None
        """
        exists, record = self.check_document_exists(document_hash)
        return record if exists else None
    
    def get_case_documents(self, case_name: str) -> List[Dict]:
        """Get all unique documents for a specific case
        
        Args:
            case_name: Name of the case
            
        Returns:
            List of document records
        """
        try:
            # Get primary documents for this case
            primary_result = self.supabase.table("document_registry").select("*").eq(
                "case_name", case_name
            ).execute()
            
            # Also check duplicate locations
            all_docs = primary_result.data or []
            
            # Get documents where this case appears in duplicates
            duplicate_result = self.supabase.table("document_registry").select("*").execute()
            
            for doc in duplicate_result.data or []:
                duplicates = doc.get("duplicate_locations", [])
                if any(dup["case_name"] == case_name for dup in duplicates):
                    # Don't add if already in list
                    if not any(d["document_hash"] == doc["document_hash"] for d in all_docs):
                        all_docs.append(doc)
            
            logger.info(f"Found {len(all_docs)} unique documents for case: {case_name}")
            return all_docs
            
        except APIError as e:
            logger.error(f"Error getting case documents: {str(e)}")
            # Return empty list instead of crashing
            return []
    

    def get_statistics(self) -> Dict:
        """Get deduplication statistics"""
        try:
            result = self.supabase.table("document_registry").select("*").execute()
            
            total_docs = len(result.data or [])
            total_duplicates = sum(
                len(doc.get("duplicate_locations", []))
                for doc in result.data or []
            )
            cases = set()
            for doc in result.data or []:
                cases.add(doc["case_name"])
                for dup in doc.get("duplicate_locations", []):
                    cases.add(dup["case_name"])
            return {
                "total_unique_documents": total_docs,
                "total_duplicate_instances": total_duplicates,  
                "total_cases": len(cases),
                "average_duplicates_per_document": (
                    total_duplicates / total_docs if total_docs > 0 else 0
                )
            }
        except APIError as e:
            logger.error(f"Error getting statistics: {e}")
            # Return empty stats instead of crashing
            return {
                "total_unique_documents": 0,
                "total_duplicate_instances": 0,
                "total_cases": 0,
                "average_duplicates_per_document": 0
            }

    async def get_statistics_async(self) -> Dict:
        """Async version of get_statistics (use in async context)"""
        try:
            # Create a new client with proper headers for async operation
            supabase_url = settings.database.supabase_url
            supabase_key = settings.database.supabase_service_key
            async_client: AsyncClient = await acreate_client(supabase_url, supabase_key)
            
            # Properly await the execute() method
            result = await async_client.table("document_registry").select("*").execute()
            
            # Close the async client connection
            await async_client.aclose()
            
            total_docs = len(result.data or [])
            total_duplicates = sum(
                len(doc.get("duplicate_locations", []))
                for doc in result.data or []
            )
            cases = set()
            for doc in result.data or []:
                cases.add(doc["case_name"])
                for dup in doc.get("duplicate_locations", []):
                    cases.add(dup["case_name"])
            return {
                "total_unique_documents": total_docs,
                "total_duplicate_instances": total_duplicates,
                "total_cases": len(cases),
                "average_duplicates_per_document": (
                    total_duplicates / total_docs if total_docs > 0 else 0
                )
            }
        except APIError as e:
            logger.error(f"Error getting statistics (async): {str(e)}")
            # Return empty stats on error
            return {
                "total_unique_documents": 0,
                "total_duplicate_instances": 0,
                "total_cases": 0,
                "average_duplicates_per_document": 0
            }