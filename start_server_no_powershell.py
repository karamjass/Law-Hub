#!/usr/bin/env python3
"""
LawHub Server Starter - PowerShell-Free Version
This script starts the Flask server without using PowerShell to avoid the -1073741510 crash.
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

def check_python():
    """Check if Python is available"""
    try:
        result = subprocess.run([sys.executable, '--version'], 
                              capture_output=True, text=True, timeout=10)
        print(f"✅ Python found: {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"❌ Python not found: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['flask', 'flask_cors', 'pandas', 'scikit-learn', 'datasets']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True

def start_flask_server():
    """Start the Flask server"""
    try:
        print("\n🚀 Starting LawHub Flask Server...")
        print("📍 Server will be available at: http://localhost:5000")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Start the Flask app
        subprocess.run([sys.executable, 'app.py'], check=True)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting server: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

def open_browser():
    """Open browser after a short delay"""
    time.sleep(3)  # Wait for server to start
    try:
        webbrowser.open('http://localhost:5000')
        print("🌐 Browser opened automatically")
    except:
        print("🌐 Please open your browser and go to: http://localhost:5000")

def main():
    """Main function"""
    print("=" * 60)
    print("           LawHub Server Starter")
    print("     PowerShell-Free Version")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path('app.py').exists():
        print("❌ Error: app.py not found in current directory")
        print("Please run this script from the LawHub project directory")
        input("Press Enter to exit...")
        return
    
    # Check Python
    if not check_python():
        input("Press Enter to exit...")
        return
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        input("Press Enter to exit...")
        return
    
    # Start browser thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start Flask server
    start_flask_server()

if __name__ == "__main__":
    main() 