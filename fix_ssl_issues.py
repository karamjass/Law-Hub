#!/usr/bin/env python3
"""
SSL Connection Fix Script for LawHub DeepSeek Integration
This script helps diagnose and fix SSL connection issues.
"""

import requests
import urllib3
import ssl
import sys
import os

def test_basic_connection():
    """Test basic internet connectivity"""
    print("🌐 Testing basic internet connectivity...")
    try:
        response = requests.get("https://www.google.com", timeout=10)
        print("✅ Basic internet connection: OK")
        return True
    except Exception as e:
        print(f"❌ Basic internet connection failed: {e}")
        return False

def test_deepseek_ssl():
    """Test DeepSeek SSL connection with different methods"""
    print("\n🔒 Testing DeepSeek SSL connection...")
    
    # Disable SSL warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Test URLs
    test_urls = [
        "https://api.deepseek.com/v1/chat/completions",
        "https://api.deepseek.com"
    ]
    
    for url in test_urls:
        print(f"\nTesting: {url}")
        
        # Method 1: Standard SSL
        try:
            response = requests.get(url, timeout=10)
            print(f"✅ Standard SSL: OK (Status: {response.status_code})")
        except requests.exceptions.SSLError as e:
            print(f"❌ Standard SSL failed: {e}")
            
            # Method 2: Disabled SSL verification
            try:
                response = requests.get(url, timeout=10, verify=False)
                print(f"✅ Disabled SSL verification: OK (Status: {response.status_code})")
            except Exception as e2:
                print(f"❌ Disabled SSL also failed: {e2}")
                
                # Method 3: Custom SSL context
                try:
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    
                    response = requests.get(url, timeout=10, verify=False)
                    print(f"✅ Custom SSL context: OK (Status: {response.status_code})")
                except Exception as e3:
                    print(f"❌ Custom SSL context failed: {e3}")

def test_api_key():
    """Test if API key is configured"""
    print("\n🔑 Testing API key configuration...")
    
    try:
        from config import DEEPSEEK_API_KEY
        if DEEPSEEK_API_KEY:
            print("✅ API key is configured")
            print(f"   Key starts with: {DEEPSEEK_API_KEY[:10]}...")
            return True
        else:
            print("❌ API key is empty")
            return False
    except ImportError:
        print("❌ config.py not found")
        return False
    except Exception as e:
        print(f"❌ Error reading API key: {e}")
        return False

def test_deepseek_api():
    """Test actual DeepSeek API call"""
    print("\n🤖 Testing DeepSeek API call...")
    
    try:
        from config import DEEPSEEK_API_KEY
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
                    "content": "Hello, this is a test."
                }
            ],
            "max_tokens": 50
        }
        
        # Try with SSL verification disabled
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30,
            verify=False
        )
        
        if response.status_code == 200:
            print("✅ DeepSeek API call successful!")
            return True
        else:
            print(f"❌ DeepSeek API call failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ DeepSeek API call error: {e}")
        return False

def suggest_fixes():
    """Suggest fixes for common issues"""
    print("\n🔧 Suggested fixes:")
    print("1. Check your internet connection")
    print("2. Try using a VPN if you're behind a corporate firewall")
    print("3. Update your Python and requests library:")
    print("   pip install --upgrade requests urllib3")
    print("4. If using a proxy, configure it properly")
    print("5. Try running the script with administrator privileges")
    print("6. Check if your antivirus/firewall is blocking the connection")

def main():
    """Main function"""
    print("🔧 LawHub SSL Connection Troubleshooter")
    print("=" * 50)
    
    # Test basic connectivity
    if not test_basic_connection():
        print("\n❌ Basic internet connectivity failed. Check your internet connection.")
        suggest_fixes()
        return
    
    # Test SSL connections
    test_deepseek_ssl()
    
    # Test API key
    if not test_api_key():
        print("\n❌ API key not configured properly.")
        print("Run: python setup_deepseek.py")
        return
    
    # Test actual API call
    if test_deepseek_api():
        print("\n🎉 All tests passed! DeepSeek integration should work.")
    else:
        print("\n❌ DeepSeek API test failed.")
        suggest_fixes()

if __name__ == "__main__":
    main() 