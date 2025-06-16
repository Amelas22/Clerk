#!/usr/bin/env python
"""
Example script for processing case documents.
Usage: python scripts/process_case.py --folder-id YOUR_FOLDER_ID [--max-docs 10]
"""

import argparse
import logging
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_injector import DocumentInjector

def setup_logging(verbose: bool = False):
    """Configure logging for the script"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Log filename with timestamp
    log_filename = f"logs/processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_filename

def main():
    """Main processing function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process legal case documents from Box to vector storage"
    )
    parser.add_argument(
        "--folder-id",
        required=True,
        help="Box folder ID to process"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to process (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test connections only, don't process documents"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Clerk Document Injector")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)
    
    try:
        # Initialize injector
        logger.info("Initializing document injector...")
        injector = DocumentInjector()
        
        # Test connections
        logger.info("Testing connections...")
        if not injector.box_client.check_connection():
            logger.error("Box connection failed!")
            return 1
        
        logger.info("✓ All connections successful")
        
        if args.dry_run:
            logger.info("Dry run mode - exiting without processing")
            return 0
        
        # Start processing
        logger.info(f"Starting processing for folder: {args.folder_id}")
        if args.max_docs:
            logger.info(f"Limited to {args.max_docs} documents")
        
        start_time = datetime.now()
        
        # Process the folder
        results = injector.process_case_folder(
            args.folder_id,
            max_documents=args.max_docs
        )
        
        # Calculate statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        successful = sum(1 for r in results if r.status == "success")
        duplicates = sum(1 for r in results if r.status == "duplicate")
        failed = sum(1 for r in results if r.status == "failed")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        logger.info(f"Documents processed: {len(results)}")
        logger.info(f"  - Successful: {successful}")
        logger.info(f"  - Duplicates: {duplicates}")
        logger.info(f"  - Failed: {failed}")
        
        if failed > 0:
            logger.info("\nFailed documents:")
            for result in results:
                if result.status == "failed":
                    logger.error(f"  - {result.file_name}: {result.error_message}")
        
        # Show case statistics
        if successful > 0 and results:
            case_name = results[0].case_name
            case_stats = injector.vector_store.get_case_statistics(case_name)
            logger.info(f"\nCase '{case_name}' statistics:")
            logger.info(f"  - Total chunks: {case_stats['total_chunks']}")
            logger.info(f"  - Total documents: {case_stats['total_documents']}")
        
        logger.info(f"\nFull log available at: {log_file}")
        
        return 0 if failed == 0 else 1
        
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())