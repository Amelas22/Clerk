#!/usr/bin/env python
"""
Cleanup script to remove Supabase references from the project.
This helps ensure a clean transition to Qdrant-only implementation.

Usage: python scripts/cleanup_supabase.py [--dry-run]
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple

# Patterns to search for Supabase references
PATTERNS = [
    r'supabase',
    r'SUPABASE',
    r'pgvector',
    r'from supabase import',
    r'create_client.*supabase',
    r'database\.supabase',
    r'settings\.database\.'
]

# Files to skip
SKIP_FILES = [
    'cleanup_supabase.py',  # This file
    'MIGRATION_GUIDE.md',    # Migration guide needs the references
    '.git',                  # Git directory
    '__pycache__',          # Python cache
    'venv',                 # Virtual environment
    'env',                  # Virtual environment
    '.env.backup',          # Backup files
]

# File extensions to check
CHECK_EXTENSIONS = [
    '.py', '.yml', '.yaml', '.md', '.txt', '.sh', '.env', '.example'
]


def find_supabase_references(root_dir: str, dry_run: bool = True) -> List[Tuple[str, int, str]]:
    """Find all Supabase references in the project
    
    Args:
        root_dir: Root directory to search
        dry_run: If True, only report findings without changes
        
    Returns:
        List of (file_path, line_number, line_content) tuples
    """
    references = []
    root_path = Path(root_dir)
    
    for file_path in root_path.rglob("*"):
        # Skip directories and binary files
        if file_path.is_dir():
            continue
            
        # Skip excluded files/directories
        if any(skip in str(file_path) for skip in SKIP_FILES):
            continue
            
        # Check file extension
        if not any(str(file_path).endswith(ext) for ext in CHECK_EXTENSIONS):
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                for pattern in PATTERNS:
                    if re.search(pattern, line, re.IGNORECASE):
                        references.append((str(file_path), i + 1, line.strip()))
                        break
                        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return references


def remove_supabase_files(root_dir: str, dry_run: bool = True) -> List[str]:
    """Remove Supabase-specific files
    
    Args:
        root_dir: Root directory
        dry_run: If True, only report what would be deleted
        
    Returns:
        List of removed files
    """
    supabase_files = [
        'src/document_processing/deduplicator.py',  # Old Supabase deduplicator
        'src/vector_storage/vector_store.py',       # Old Supabase vector store
        'src/vector_storage/fulltext_search.py',    # Old full-text search
        'migrations/001_hybrid_search_setup.sql',   # Supabase migrations
        'supabase/',                                # Supabase directory
    ]
    
    removed = []
    root_path = Path(root_dir)
    
    for file_pattern in supabase_files:
        file_path = root_path / file_pattern
        
        if file_path.exists():
            if dry_run:
                print(f"[DRY RUN] Would remove: {file_path}")
            else:
                if file_path.is_dir():
                    import shutil
                    shutil.rmtree(file_path)
                else:
                    file_path.unlink()
                print(f"Removed: {file_path}")
            removed.append(str(file_path))
    
    return removed


def update_docker_compose(root_dir: str, dry_run: bool = True) -> bool:
    """Update docker-compose.yml to remove Supabase include
    
    Args:
        root_dir: Root directory
        dry_run: If True, only report changes
        
    Returns:
        True if updated successfully
    """
    compose_file = Path(root_dir) / 'docker-compose.yml'
    
    if not compose_file.exists():
        print(f"docker-compose.yml not found at {compose_file}")
        return False
    
    try:
        with open(compose_file, 'r') as f:
            content = f.read()
        
        # Check if already updated
        if '# include:' in content and '# - ./supabase/docker/docker-compose.yml' in content:
            print("docker-compose.yml already updated")
            return True
        
        # Comment out Supabase include
        updated_content = re.sub(
            r'^include:\n\s*- \./supabase/docker/docker-compose\.yml',
            '# include:\n#   - ./supabase/docker/docker-compose.yml',
            content,
            flags=re.MULTILINE
        )
        
        if dry_run:
            print("[DRY RUN] Would update docker-compose.yml")
        else:
            with open(compose_file, 'w') as f:
                f.write(updated_content)
            print("Updated docker-compose.yml")
        
        return True
        
    except Exception as e:
        print(f"Error updating docker-compose.yml: {e}")
        return False


def create_backup_env(root_dir: str) -> bool:
    """Create backup of .env file before removing Supabase variables
    
    Args:
        root_dir: Root directory
        
    Returns:
        True if backup created
    """
    env_file = Path(root_dir) / '.env'
    backup_file = Path(root_dir) / '.env.backup_supabase'
    
    if env_file.exists() and not backup_file.exists():
        try:
            import shutil
            shutil.copy2(env_file, backup_file)
            print(f"Created .env backup at {backup_file}")
            return True
        except Exception as e:
            print(f"Error creating backup: {e}")
            return False
    
    return False


def main():
    """Main cleanup function"""
    parser = argparse.ArgumentParser(
        description="Clean up Supabase references from the project"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be changed without making changes"
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory of the project (default: current directory)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SUPABASE CLEANUP SCRIPT")
    print("="*60)
    
    if args.dry_run:
        print("\n[DRY RUN MODE] No changes will be made\n")
    
    # Create backup of .env
    if not args.dry_run:
        create_backup_env(args.root)
    
    # Find Supabase references
    print("\n1. Searching for Supabase references...")
    references = find_supabase_references(args.root, args.dry_run)
    
    if references:
        print(f"\nFound {len(references)} Supabase references:")
        
        # Group by file
        by_file = {}
        for file_path, line_num, content in references:
            if file_path not in by_file:
                by_file[file_path] = []
            by_file[file_path].append((line_num, content))
        
        for file_path, lines in by_file.items():
            print(f"\n{file_path}:")
            for line_num, content in lines[:5]:  # Show first 5
                print(f"  Line {line_num}: {content[:80]}...")
            if len(lines) > 5:
                print(f"  ... and {len(lines) - 5} more")
    else:
        print("No Supabase references found!")
    
    # Remove Supabase files
    print("\n2. Removing Supabase-specific files...")
    removed_files = remove_supabase_files(args.root, args.dry_run)
    if removed_files:
        print(f"Removed {len(removed_files)} files")
    else:
        print("No Supabase-specific files found")
    
    # Update docker-compose.yml
    print("\n3. Updating docker-compose.yml...")
    update_docker_compose(args.root, args.dry_run)
    
    # Final recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. Update your .env file to remove:")
    print("   - SUPABASE_URL")
    print("   - SUPABASE_KEY") 
    print("   - SUPABASE_SERVICE_KEY")
    
    print("\n2. Update any custom code that imports:")
    print("   - from supabase import ...")
    print("   - from src.document_processing.deduplicator import ...")
    
    print("\n3. Stop and remove Supabase containers:")
    print("   docker-compose down")
    print("   docker volume prune  # Remove unused volumes")
    
    print("\n4. Start fresh with Qdrant-only setup:")
    print("   docker-compose up -d")
    print("   python scripts/setup_qdrant.py")
    
    if not args.dry_run:
        print("\n✓ Cleanup complete! Your project now uses Qdrant exclusively.")
    else:
        print("\n[DRY RUN] Run without --dry-run to apply changes")


if __name__ == "__main__":
    main()