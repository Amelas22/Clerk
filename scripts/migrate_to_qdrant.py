#!/usr/bin/env python
"""
This migration script is no longer needed.
All database operations now use Qdrant exclusively.

The Clerk legal AI system has been updated to use Qdrant for:
- Vector storage and search
- Document deduplication
- Metadata management

No migration from Supabase is required.
"""

import sys

def main():
    """Inform users that migration is not needed"""
    print("\n" + "="*80)
    print("MIGRATION NOT REQUIRED")
    print("="*80)
    print("\nThe Clerk legal AI system now uses Qdrant exclusively.")
    print("No migration from Supabase is needed.\n")
    print("To set up the system:")
    print("1. Ensure Qdrant is running: docker-compose up -d qdrant")
    print("2. Run setup script: python scripts/setup_qdrant.py")
    print("3. Start processing: python -m src.document_injector --folder-id YOUR_FOLDER")
    print("\nFor more information, see README_QDRANT.md")
    print("="*80 + "\n")
    
    sys.exit(0)

if __name__ == "__main__":
    main()


class SupabaseToQdrantMigrator:
    """Handles migration from Supabase pgvector to Qdrant"""
    
    def __init__(self, dry_run: bool = False):
        """Initialize migrator
        
        Args:
            dry_run: If True, only analyze without migrating
        """
        self.dry_run = dry_run
        
        # Initialize Supabase client
        self.supabase = create_client(
            settings.database.supabase_url,
            settings.database.supabase_service_key
        )
        
        # Initialize Qdrant
        self.qdrant_store = QdrantVectorStore()
        self.sparse_encoder = SparseVectorEncoder()
        
        # Migration statistics
        self.stats = {
            "total_vectors": 0,
            "migrated": 0,
            "failed": 0,
            "cases_processed": set(),
            "start_time": None,
            "end_time": None
        }
    
    def analyze_supabase_data(self) -> Dict[str, Any]:
        """Analyze existing Supabase data
        
        Returns:
            Analysis results
        """
        logger.info("Analyzing Supabase vector data...")
        
        try:
            # Get total count
            count_result = self.supabase.table("case_documents").select(
                "id", count="exact"
            ).execute()
            
            total_count = count_result.count
            
            # Get case distribution
            case_result = self.supabase.table("case_documents").select(
                "case_name"
            ).execute()
            
            cases = {}
            for row in case_result.data:
                case_name = row["case_name"]
                cases[case_name] = cases.get(case_name, 0) + 1
            
            # Get sample data for size estimation
            sample_result = self.supabase.table("case_documents").select(
                "*"
            ).limit(100).execute()
            
            avg_content_size = 0
            if sample_result.data:
                total_size = sum(len(row.get("content", "")) for row in sample_result.data)
                avg_content_size = total_size / len(sample_result.data)
            
            analysis = {
                "total_vectors": total_count,
                "unique_cases": len(cases),
                "case_distribution": cases,
                "average_content_size": avg_content_size,
                "estimated_memory_gb": (total_count * 1536 * 4) / (1024**3),  # Float32
                "estimated_memory_gb_quantized": (total_count * 1536) / (1024**3),  # Int8
            }
            
            # Display analysis
            logger.info(f"Total vectors: {analysis['total_vectors']:,}")
            logger.info(f"Unique cases: {analysis['unique_cases']}")
            logger.info(f"Estimated memory: {analysis['estimated_memory_gb']:.2f} GB")
            logger.info(f"With quantization: {analysis['estimated_memory_gb_quantized']:.2f} GB")
            
            logger.info("\nCase distribution:")
            for case, count in sorted(cases.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"  {case}: {count:,} vectors")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Supabase data: {str(e)}")
            raise
    
    def migrate_case(self, case_name: str, batch_size: int = 1000) -> int:
        """Migrate a single case to Qdrant
        
        Args:
            case_name: Case to migrate
            batch_size: Batch size for processing
            
        Returns:
            Number of vectors migrated
        """
        logger.info(f"Migrating case: {case_name}")
        migrated = 0
        
        try:
            # Get all vectors for this case
            offset = 0
            
            while True:
                # Fetch batch from Supabase
                result = self.supabase.table("case_documents").select(
                    "*"
                ).eq(
                    "case_name", case_name
                ).range(
                    offset, offset + batch_size - 1
                ).execute()
                
                if not result.data:
                    break
                
                # Prepare Qdrant points
                chunks = []
                
                for row in result.data:
                    # Generate sparse vectors
                    keyword_sparse, citation_sparse = self.sparse_encoder.encode_for_hybrid_search(
                        row["content"]
                    )
                    
                    # Prepare chunk data
                    chunk = {
                        "content": row["content"],
                        "embedding": row["embedding"],
                        "search_text": row.get("search_text", row["content"]),
                        "keywords_sparse": keyword_sparse,
                        "citations_sparse": citation_sparse,
                        "metadata": row.get("metadata", {})
                    }
                    
                    # Ensure case isolation
                    chunk["metadata"]["case_name"] = case_name
                    chunk["metadata"]["document_id"] = row.get("document_id", "")
                    chunk["metadata"]["migrated_from_supabase"] = True
                    chunk["metadata"]["migration_date"] = datetime.utcnow().isoformat()
                    
                    chunks.append(chunk)
                
                # Store in Qdrant
                if not self.dry_run:
                    chunk_ids = self.qdrant_store.store_document_chunks(
                        case_name=case_name,
                        document_id=f"migration_{case_name}_{offset}",
                        chunks=chunks,
                        use_hybrid=True
                    )
                    migrated += len(chunk_ids)
                else:
                    logger.info(f"[DRY RUN] Would migrate {len(chunks)} chunks")
                    migrated += len(chunks)
                
                offset += batch_size
                
                # Progress update
                if offset % 10000 == 0:
                    logger.info(f"  Processed {offset:,} vectors for {case_name}")
            
            logger.info(f"Completed migration for {case_name}: {migrated:,} vectors")
            return migrated
            
        except Exception as e:
            logger.error(f"Error migrating case {case_name}: {str(e)}")
            raise
    
    def verify_migration(self, case_name: str) -> Dict[str, Any]:
        """Verify migration for a case
        
        Args:
            case_name: Case to verify
            
        Returns:
            Verification results
        """
        try:
            # Get count from Supabase
            supabase_result = self.supabase.table("case_documents").select(
                "id", count="exact"
            ).eq("case_name", case_name).execute()
            
            supabase_count = supabase_result.count
            
            # Get count from Qdrant
            qdrant_stats = self.qdrant_store.get_case_statistics(case_name)
            qdrant_count = qdrant_stats["total_chunks"]
            
            # Compare
            match = supabase_count == qdrant_count
            
            verification = {
                "case_name": case_name,
                "supabase_count": supabase_count,
                "qdrant_count": qdrant_count,
                "match": match,
                "difference": abs(supabase_count - qdrant_count)
            }
            
            if match:
                logger.info(f"✓ Verification passed for {case_name}: {supabase_count:,} vectors")
            else:
                logger.warning(
                    f"✗ Verification failed for {case_name}: "
                    f"Supabase: {supabase_count:,}, Qdrant: {qdrant_count:,}"
                )
            
            return verification
            
        except Exception as e:
            logger.error(f"Error verifying case {case_name}: {str(e)}")
            return {
                "case_name": case_name,
                "error": str(e)
            }
    
    async def migrate_all(self, batch_size: int = 1000, 
                         max_concurrent: int = 3) -> Dict[str, Any]:
        """Migrate all cases
        
        Args:
            batch_size: Batch size per case
            max_concurrent: Maximum concurrent migrations
            
        Returns:
            Migration summary
        """
        self.stats["start_time"] = datetime.utcnow()
        
        # Analyze data first
        analysis = self.analyze_supabase_data()
        self.stats["total_vectors"] = analysis["total_vectors"]
        
        if self.dry_run:
            logger.info("\n[DRY RUN MODE] No data will be migrated")
        
        # Get all cases
        cases = list(analysis["case_distribution"].keys())
        logger.info(f"\nMigrating {len(cases)} cases...")
        
        # Process cases
        for case in cases:
            try:
                migrated = self.migrate_case(case, batch_size)
                self.stats["migrated"] += migrated
                self.stats["cases_processed"].add(case)
            except Exception as e:
                logger.error(f"Failed to migrate case {case}: {str(e)}")
                self.stats["failed"] += analysis["case_distribution"][case]
        
        self.stats["end_time"] = datetime.utcnow()
        
        # Verify migration
        if not self.dry_run:
            logger.info("\nVerifying migration...")
            verification_results = []
            
            for case in self.stats["cases_processed"]:
                result = self.verify_migration(case)
                verification_results.append(result)
            
            # Summary
            verified = sum(1 for r in verification_results if r.get("match", False))
            logger.info(f"\nVerification complete: {verified}/{len(verification_results)} cases match")
        
        return self._generate_summary()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate migration summary
        
        Returns:
            Summary dictionary
        """
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        summary = {
            "total_vectors": self.stats["total_vectors"],
            "migrated": self.stats["migrated"],
            "failed": self.stats["failed"],
            "success_rate": (self.stats["migrated"] / self.stats["total_vectors"] * 100) 
                           if self.stats["total_vectors"] > 0 else 0,
            "cases_processed": len(self.stats["cases_processed"]),
            "duration_seconds": duration,
            "vectors_per_second": self.stats["migrated"] / duration if duration > 0 else 0,
            "dry_run": self.dry_run
        }
        
        # Display summary
        logger.info("\n" + "=" * 50)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total vectors: {summary['total_vectors']:,}")
        logger.info(f"Migrated: {summary['migrated']:,}")
        logger.info(f"Failed: {summary['failed']:,}")
        logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"Cases processed: {summary['cases_processed']}")
        logger.info(f"Duration: {summary['duration_seconds']:.1f} seconds")
        logger.info(f"Speed: {summary['vectors_per_second']:.1f} vectors/second")
        
        if self.dry_run:
            logger.info("\n[DRY RUN] No data was actually migrated")
        
        return summary
    
    def rollback(self, case_name: Optional[str] = None):
        """Rollback migration by deleting migrated data from Qdrant
        
        Args:
            case_name: Specific case to rollback, or None for all
        """
        if self.dry_run:
            logger.info("[DRY RUN] Rollback not applicable")
            return
        
        logger.warning("Starting rollback...")
        
        try:
            if case_name:
                # Delete specific case
                count = self.qdrant_store.delete_case_vectors(case_name)
                logger.info(f"Deleted {count} vectors for case {case_name}")
            else:
                # Delete all migrated data
                # This would need to be implemented based on migration metadata
                logger.warning("Full rollback not implemented - delete collections manually")
        
        except Exception as e:
            logger.error(f"Error during rollback: {str(e)}")
            raise


def main():
    """Main migration function"""
    parser = argparse.ArgumentParser(
        description="Migrate vector data from Supabase to Qdrant"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for migration (default: 1000)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze without migrating data"
    )
    parser.add_argument(
        "--case",
        type=str,
        help="Migrate specific case only"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing migration"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback migration"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        migrator = SupabaseToQdrantMigrator(dry_run=args.dry_run)
        
        if args.rollback:
            migrator.rollback(args.case)
        elif args.verify_only:
            if args.case:
                migrator.verify_migration(args.case)
            else:
                logger.error("Please specify a case to verify with --case")
        elif args.case:
            # Migrate single case
            migrator.migrate_case(args.case, args.batch_size)
            if not args.dry_run:
                migrator.verify_migration(args.case)
        else:
            # Migrate all
            asyncio.run(migrator.migrate_all(args.batch_size))
        
    except KeyboardInterrupt:
        logger.info("\nMigration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()