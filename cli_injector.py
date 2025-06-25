#!/usr/bin/env python3
"""
CLI Document Injector for Clerk Legal AI System
Processes documents from Box folders into vector databases.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.document_injector import DocumentInjector
from src.utils.logger import setup_logging
from config.settings import settings


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Inject legal documents from Box into vector databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single folder
  %(prog)s --folder-id 123456789
  
  # Process with document limit
  %(prog)s --folder-id 123456789 --max-documents 10
  
  # Process all case folders from root
  %(prog)s --root 987654321 --max-folders 5
  
  # Enable debug logging
  %(prog)s --folder-id 123456789 --log-level DEBUG
        """
    )
    
    # Mutually exclusive group for folder-id vs root
    folder_group = parser.add_mutually_exclusive_group(required=True)
    folder_group.add_argument(
        '--folder-id',
        type=str,
        help='Box folder ID to process as a single case'
    )
    folder_group.add_argument(
        '--root',
        type=str,
        help='Root folder ID containing multiple case folders'
    )
    
    # Processing limits
    parser.add_argument(
        '--max-documents',
        type=int,
        default=None,
        help='Maximum number of documents to process (excludes duplicates)'
    )
    parser.add_argument(
        '--max-folders',
        type=int,
        default=None,
        help='Maximum number of case folders to process (only with --root)'
    )
    
    # Options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    parser.add_argument(
        '--skip-cost-tracking',
        action='store_true',
        help='Disable API cost tracking'
    )
    parser.add_argument(
        '--save-cost-report',
        action='store_true',
        help='Save cost report after processing'
    )
    parser.add_argument(
        '--no-context',
        action='store_true',
        help='Skip context generation for chunks'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.max_folders and not args.root:
        parser.error("--max-folders can only be used with --root")
    
    return args


def process_single_folder(injector: DocumentInjector, folder_id: str,
                       max_documents: Optional[int] = None) -> dict:
    try:
        # Process the folder
        results = injector.process_case_folder(
            folder_id=folder_id,
            max_documents=max_documents
        )
        
        # Summary
        success_count = sum(1 for r in results if r.status == "success")
        duplicate_count = sum(1 for r in results if r.status == "duplicate")
        failed_count = sum(1 for r in results if r.status == "error")
        
        logger = logging.getLogger(__name__)
        logger.info(f"Folder {folder_id} processing complete: "
                   f"{success_count} processed, "
                   f"{duplicate_count} duplicates, "
                   f"{failed_count} failed")
        
        return {
            "status": "complete",
            "folder_id": folder_id,
            "success": success_count,
            "duplicates": duplicate_count,
            "failed": failed_count
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error processing folder {folder_id}: {str(e)}")
        return {"status": "error", "folder_id": folder_id, "error": str(e)}


def process_root_folder(injector: DocumentInjector, root_id: str,
                       max_folders: Optional[int] = None,
                       max_documents: Optional[int] = None,
                       dry_run: bool = False) -> dict:
    """Process a root folder by processing each case folder within it."""
    logger = logging.getLogger(__name__)
    
    # Get the case folders under the root
    case_folders = injector.box_client.get_folder_items(root_id)
    case_folders = [f for f in case_folders if f['type'] == 'folder']
    
    if max_folders is not None:
        case_folders = case_folders[:max_folders]
    
    logger.info(f"Found {len(case_folders)} case folders under root {root_id}")
    
    if dry_run:
        logger.info("Dry run enabled, skipping actual processing")
        return {"status": "dry_run", "folders": len(case_folders)}
    
    # Process each case folder
    results = []
    for i, folder in enumerate(case_folders, 1):
        logger.info(f"\nProcessing case {i}/{len(case_folders)}: "
                   f"{folder['name']}")
        
        folder_result = process_single_folder(
            injector,
            folder_id=folder['id'],
            max_documents=max_documents
        )
        folder_result['case_name'] = folder['name']
        results.append(folder_result)
    
    # Summary
    total_success = sum(r.get('success', 0) for r in results)
    total_duplicates = sum(r.get('duplicates', 0) for r in results)
    total_failed = sum(r.get('failed', 0) for r in results)
    
    logger.info(f"Root folder processing complete: "
               f"{total_success} processed, "
               f"{total_duplicates} duplicates, "
               f"{total_failed} failed")
    
    return {
        "status": "complete",
        "root_id": root_id,
        "success": total_success,
        "duplicates": total_duplicates,
        "failed": total_failed,
        "folders": results
    }


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Configure logging with UTF-8 support
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('clerk.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("Clerk Document Injector starting...")
    logger.info(f"Configuration: {args}")
    
    # Validate settings
    if not settings.validate():
        logger.error("Invalid configuration. Please check your .env file.")
        sys.exit(1)
    
    try:
        # Initialize injector
        injector = DocumentInjector(
            enable_cost_tracking=not args.skip_cost_tracking,
            no_context=args.no_context
        )
        
        # Test connections
        logger.info("Testing connections...")
        if not injector.box_client.check_connection():
            logger.error("Failed to connect to Box API")
            sys.exit(1)
        
        logger.info("Connections verified [OK]")
        
        # Process based on mode
        if args.folder_id:
            result = process_single_folder(
                injector, args.folder_id, 
                args.max_documents
            )
        else:  # args.root
            result = process_root_folder(
                injector, args.root, args.max_folders,
                args.max_documents, args.dry_run
            )
        
        # Save cost report if requested
        if args.save_cost_report and not args.skip_cost_tracking:
            if hasattr(injector, 'cost_tracker'):
                report_path = injector.cost_tracker.save_report()
                logger.info(f"Cost report saved to: {report_path}")
        
        # Print final status
        if result['status'] == 'complete':
            logger.info("Processing completed successfully!")
            sys.exit(0)
        else:
            logger.error("Processing failed or incomplete")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()