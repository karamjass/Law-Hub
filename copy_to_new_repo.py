#!/usr/bin/env python3
"""
Copy LawHub to New Repository Location
This script copies all LawHub files to D:\LawHub without using PowerShell
"""

import os
import shutil
import sys
from pathlib import Path

def copy_lawhub_to_new_location():
    """Copy all LawHub files to the new repository location"""
    print("📁 Copying LawHub to New Repository Location")
    print("=" * 50)
    
    # Source and destination paths
    source_path = Path("C:/Users/Japneet Singh/Desktop/LawHub-1")
    dest_path = Path("C:/LawHub_Project")
    
    print(f"Source: {source_path}")
    print(f"Destination: {dest_path}")
    
    # Check if source exists
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_path}")
        return False
    
    try:
        # Create destination directory if it doesn't exist
        dest_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Destination directory created/verified: {dest_path}")
        
        # Copy all files and directories
        print("\n📋 Copying files...")
        
        # List of files and directories to copy
        items_to_copy = [
            "app.py",
            "requirements.txt",
            "README.md",
            "config.py",
            "setup_deepseek.py",
            "fix_ssl_issues.py",
            "fix_deepseek_balance.py",
            "test_deepseek.py",
            "start_server.bat",
            "start_server_no_powershell.py",
            "templates/",
            "kaggle (3).json"
        ]
        
        copied_count = 0
        for item in items_to_copy:
            source_item = source_path / item
            dest_item = dest_path / item
            
            if source_item.exists():
                if source_item.is_file():
                    # Copy file
                    shutil.copy2(source_item, dest_item)
                    print(f"✅ Copied file: {item}")
                    copied_count += 1
                elif source_item.is_dir():
                    # Copy directory
                    if dest_item.exists():
                        shutil.rmtree(dest_item)
                    shutil.copytree(source_item, dest_item)
                    print(f"✅ Copied directory: {item}")
                    copied_count += 1
            else:
                print(f"⚠️  Item not found: {item}")
        
        # Also copy any other files that might exist
        for item in source_path.iterdir():
            if item.name not in [i.split('/')[0] for i in items_to_copy if '/' in i] and item.name not in [i for i in items_to_copy if '/' not in i]:
                if item.is_file():
                    shutil.copy2(item, dest_path / item.name)
                    print(f"✅ Copied additional file: {item.name}")
                    copied_count += 1
                elif item.is_dir():
                    dest_dir = dest_path / item.name
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                    shutil.copytree(item, dest_dir)
                    print(f"✅ Copied additional directory: {item.name}")
                    copied_count += 1
        
        print(f"\n🎉 Successfully copied {copied_count} items to {dest_path}")
        
        # Verify the copy
        print("\n🔍 Verifying copy...")
        if (dest_path / "app.py").exists():
            print("✅ Main application file copied successfully")
        if (dest_path / "templates").exists():
            print("✅ Templates directory copied successfully")
        if (dest_path / "requirements.txt").exists():
            print("✅ Requirements file copied successfully")
        
        print(f"\n🚀 LawHub has been successfully copied to: {dest_path}")
        print("You can now run the application from the new location!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error copying files: {e}")
        return False

def main():
    """Main function"""
    print("LawHub Repository Copy Tool")
    print("=" * 50)
    
    if copy_lawhub_to_new_location():
        print("\n✅ Copy completed successfully!")
        print("\n📝 Next steps:")
        print("1. Navigate to C:\\LawHub_Project")
        print("2. Run: python app.py")
        print("3. Or use: python start_server_no_powershell.py")
    else:
        print("\n❌ Copy failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 