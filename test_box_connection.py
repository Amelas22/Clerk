#!/usr/bin/env python3
"""
Quick test script to verify Box API connection.
Tests authentication and basic API functionality without downloading files.
"""

import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_box_connection():
    """Test Box API connection and basic functionality"""
    
    print("🔍 Testing Box API Connection...")
    print("=" * 50)
    
    try:
        from src.document_processing.box_client import BoxClient  
        
        # Create Box client
        print("📦 Initializing Box client...")
        box_client = BoxClient()
        
        # Test basic connection
        print("🔗 Testing connection...")
        if box_client.check_connection():
            print("✅ Connection successful!")
        else:
            print("❌ Connection failed!")
            return False
        
        # Get current user info
        print("\n👤 Getting user information...")
        try:
            current_user = box_client.client.user().get()
            print(f"   User: {current_user.name}")
            print(f"   Email: {current_user.login}")
            print(f"   User ID: {current_user.id}")
        except Exception as e:
            print(f"   ⚠️  Could not get user info: {e}")
        
        # Test folder access (using root folder)
        print("\n📁 Testing folder access...")
        try:
            root_folder = box_client.client.folder('0').get()
            print(f"   Root folder: {root_folder.name}")
            print(f"   Folder ID: {root_folder.id}")
            
            # Get a few items from root (just to test API calls)
            items = list(root_folder.get_items(limit=5))
            print(f"   Found {len(items)} items in root folder")
            
            if items:
                print("   📋 Sample items:")
                for item in items[:3]:  # Show first 3 items
                    print(f"      - {item.name} ({item.type})")
            
        except Exception as e:
            print(f"   ⚠️  Could not access folders: {e}")
        
        # Test enterprise info
        print("\n🏢 Testing enterprise access...")
        try:
            enterprise = box_client.client.get_current_enterprise()
            if enterprise:
                print(f"   Enterprise: {enterprise.name}")
                print(f"   Enterprise ID: {enterprise.id}")
            else:
                print("   ℹ️  No enterprise access or personal account")
        except Exception as e:
            print(f"   ⚠️  Could not get enterprise info: {e}")
        
        print("\n🎉 Box API test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure your BoxClient module is in the Python path")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Full error details:")
        return False

def test_environment_variables():
    """Check if required environment variables are set"""
    
    print("\n🔧 Checking environment variables...")
    print("-" * 30)
    
    import os
    
    required_vars = [
        'BOX_CLIENT_ID',
        'BOX_CLIENT_SECRET', 
        'BOX_ENTERPRISE_ID',
        'BOX_JWT_KEY_ID',
        'BOX_PRIVATE_KEY',
        'BOX_PASSPHRASE'
    ]
    
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Show first few characters for verification (without exposing secrets)
            display_value = value[:8] + "..." if len(value) > 8 else value
            if var == 'BOX_PRIVATE_KEY':
                display_value = "-----BEGIN..." if value.startswith('-----BEGIN') else "Invalid format"
            print(f"   ✅ {var}: {display_value}")
        else:
            print(f"   ❌ {var}: Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Missing variables: {', '.join(missing_vars)}")
        return False
    else:
        print("\n✅ All environment variables are set!")
        return True

def main():
    """Main test function"""
    
    print(f"🚀 Box API Connection Test")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Python: {sys.version}")
    
    # Check environment variables first
    env_ok = test_environment_variables()
    
    if not env_ok:
        print("\n❌ Environment check failed. Please set required variables.")
        sys.exit(1)
    
    # Test Box connection
    connection_ok = test_box_connection()
    
    if connection_ok:
        print("\n🎯 All tests passed! Box API is ready to use.")
        sys.exit(0)
    else:
        print("\n💥 Tests failed. Check your configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()