#!/usr/bin/env python3
"""
DeepSeek Balance Fix Script for LawHub
This script helps diagnose and fix DeepSeek API balance issues.
"""

import requests
import urllib3
import json

def check_deepseek_balance():
    """Check DeepSeek API balance and account status"""
    print("💰 DeepSeek Account Balance Checker")
    print("=" * 50)
    
    try:
        from config import DEEPSEEK_API_KEY
        if not DEEPSEEK_API_KEY:
            print("❌ No API key configured")
            print("Run: python setup_deepseek.py")
            return False
        
        print(f"✅ API key found: {DEEPSEEK_API_KEY[:10]}...")
        
        # Test API call to check balance
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ],
            "max_tokens": 10
        }
        
        print("\n🧪 Testing API call...")
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30,
            verify=False
        )
        
        if response.status_code == 200:
            print("✅ API call successful! Your account has sufficient balance.")
            return True
        elif response.status_code == 402:
            print("❌ Insufficient Balance Error (402)")
            print("Your DeepSeek account needs more credits.")
            suggest_balance_fixes()
            return False
        elif response.status_code == 401:
            print("❌ Unauthorized Error (401)")
            print("Your API key might be invalid or expired.")
            suggest_api_key_fixes()
            return False
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def suggest_balance_fixes():
    """Suggest fixes for balance issues"""
    print("\n💡 How to Fix Insufficient Balance:")
    print("1. Visit: https://platform.deepseek.com/")
    print("2. Log in to your account")
    print("3. Go to Billing/Credits section")
    print("4. Add credits to your account")
    print("5. Wait a few minutes for the balance to update")
    print("\n💡 Alternative: Use the '📋 Legal QA' button instead")
    print("   This uses your local legal database and doesn't require credits.")

def suggest_api_key_fixes():
    """Suggest fixes for API key issues"""
    print("\n💡 How to Fix API Key Issues:")
    print("1. Visit: https://platform.deepseek.com/")
    print("2. Go to API Keys section")
    print("3. Create a new API key")
    print("4. Update your config.py with the new key")
    print("5. Or run: python setup_deepseek.py")

def test_fallback_system():
    """Test the fallback system"""
    print("\n🔄 Testing Fallback System...")
    
    try:
        # Test the fallback endpoint
        response = requests.post(
            'http://localhost:5000/api/deepseek_legal',
            headers={'Content-Type': 'application/json'},
            json={'question': 'What are my legal rights?'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('fallback'):
                print("✅ Fallback system working!")
                print("   When DeepSeek is unavailable, it will use your legal database.")
            else:
                print("✅ DeepSeek API working!")
        else:
            print(f"❌ Fallback test failed: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Start with: python app.py")
    except Exception as e:
        print(f"❌ Fallback test error: {e}")

def main():
    """Main function"""
    print("🔧 DeepSeek Balance Fixer for LawHub")
    print("=" * 50)
    
    # Check balance
    if check_deepseek_balance():
        print("\n🎉 DeepSeek is working properly!")
        print("You can use the 🤖 DeepSeek AI button in the app.")
    else:
        print("\n⚠️ DeepSeek has issues, but you can still use:")
        print("   - 📋 Legal QA button (uses your legal database)")
        print("   - 📚 Laws Explorer")
        print("   - 📄 Document Analysis")
        print("   - 🚨 Emergency Help")
    
    # Test fallback if server is running
    print("\n" + "=" * 50)
    test_fallback_system()

if __name__ == "__main__":
    main() 