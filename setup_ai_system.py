#!/usr/bin/env python3
"""
Setup Script for AI Property Intelligence System
Installs dependencies and tests the system
"""

import subprocess
import sys
import os
import json

def install_requirements():
    """Install required packages"""
    requirements = [
        'aiohttp',
        'requests',
        'asyncio'
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def check_openai_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✅ OpenAI API key found")
        return True
    else:
        print("❌ OpenAI API key not found")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False

def test_system():
    """Run a basic test of the AI system"""
    print("\n🧪 Testing AI Property Intelligence System...")
    
    if not check_openai_key():
        return False
        
    # Check if test image exists
    test_image = "uploads/lot2_324_dolan_road.png"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        print("Please ensure the LOT 2 map image is available for testing")
        return False
    
    print("✅ Test image found")
    return True

def main():
    print("🚀 Setting up AI Property Intelligence System")
    print("=" * 50)
    
    # Install requirements
    print("📦 Installing required packages...")
    install_requirements()
    
    # Check setup
    print("\n🔧 Checking system setup...")
    if test_system():
        print("\n✅ System setup complete!")
        print("\nNext steps:")
        print("1. Ensure your OpenAI API key is set")
        print("2. Run: python ai_property_intelligence_v1.py")
        print("3. The system will analyze the LOT 2 map and extract coordinates")
    else:
        print("\n❌ Setup incomplete. Please fix the issues above.")

if __name__ == "__main__":
    main() 