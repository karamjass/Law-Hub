#!/usr/bin/env python3
"""
DeepSeek AI Setup Script for LawHub
This script helps you configure your DeepSeek API key.
"""

import os
import sys

def setup_deepseek():
    print("🤖 DeepSeek AI Setup for LawHub")
    print("=" * 40)
    
    # Check if config.py exists
    if os.path.exists('config.py'):
        print("✅ Configuration file found")
    else:
        print("❌ Configuration file not found. Creating one...")
        create_config_file()
    
    # Get API key from user
    print("\n📝 Please enter your DeepSeek API key:")
    print("   (Get it from: https://platform.deepseek.com/)")
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Setup cancelled.")
        return False
    
    # Update config file
    try:
        with open('config.py', 'r') as f:
            content = f.read()
        
        # Replace the API key
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('DEEPSEEK_API_KEY = ""'):
                lines[i] = f'DEEPSEEK_API_KEY = "{api_key}"'
                break
        
        with open('config.py', 'w') as f:
            f.write('\n'.join(lines))
        
        print("✅ API key configured successfully!")
        print("🚀 You can now start the server with: python app.py")
        return True
        
    except Exception as e:
        print(f"❌ Error updating configuration: {e}")
        return False

def create_config_file():
    """Create a default config file"""
    config_content = '''# LawHub Configuration File
# Set your DeepSeek API key here

# DeepSeek AI Configuration
DEEPSEEK_API_KEY = ""  # Add your DeepSeek API key here

# Example:
# DEEPSEEK_API_KEY = "sk-your-api-key-here"

# To get your DeepSeek API key:
# 1. Visit https://platform.deepseek.com/
# 2. Sign up or log in to your account
# 3. Go to API Keys section
# 4. Create a new API key
# 5. Copy the key and paste it above

# Alternative: Set as environment variable
# export DEEPSEEK_API_KEY="sk-your-api-key-here"

# Server Configuration
FLASK_HOST = "localhost"
FLASK_PORT = 5000
FLASK_DEBUG = True
'''
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    print("✅ Configuration file created")

def test_connection():
    """Test the DeepSeek connection"""
    print("\n🧪 Testing DeepSeek connection...")
    
    try:
        from config import DEEPSEEK_API_KEY
        import requests
        import urllib3
        
        # Disable SSL warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        if not DEEPSEEK_API_KEY:
            print("❌ No API key configured")
            return False
        
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, this is a test message."
                }
            ],
            "max_tokens": 50
        }
        
        # Try with SSL verification disabled to handle SSL issues
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30,
            verify=False  # Disable SSL verification
        )
        
        if response.status_code == 200:
            print("✅ DeepSeek connection successful!")
            return True
        else:
            print(f"❌ Connection failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.SSLError as e:
        print(f"❌ SSL Connection error: {e}")
        print("💡 Try running: python fix_ssl_issues.py")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("💡 Try running: python fix_ssl_issues.py")
        return False

if __name__ == "__main__":
    print("Welcome to LawHub DeepSeek AI Setup!")
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_connection()
    else:
        if setup_deepseek():
            print("\n🔍 Would you like to test the connection? (y/n): ", end="")
            if input().lower().startswith('y'):
                test_connection() 