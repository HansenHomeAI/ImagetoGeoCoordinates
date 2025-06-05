#!/usr/bin/env python3
"""
Setup script for ImagetoGeoCoordinates application
This script helps with installing dependencies and setting up the environment
"""

import subprocess
import sys
import os

def install_requirements():
    """Install Python requirements"""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Python dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        return False
    return True

def check_system_dependencies():
    """Check for system dependencies"""
    print("Checking system dependencies...")
    
    # Check for Tesseract
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
        print("✅ Tesseract OCR found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Tesseract OCR not found. Please install it:")
        print("   macOS: brew install tesseract")
        print("   Ubuntu: sudo apt-get install tesseract-ocr")
        print("   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False
    
    # Check for poppler-utils (for PDF processing)
    try:
        subprocess.run(["pdftoppm", "-h"], capture_output=True, check=True)
        print("✅ Poppler utils found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Poppler utils not found. Please install it:")
        print("   macOS: brew install poppler")
        print("   Ubuntu: sudo apt-get install poppler-utils")
        print("   Windows: Download from https://poppler.freedesktop.org/")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'processed', 'static', 'templates']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")

def run_tests():
    """Run basic functionality tests"""
    print("Running basic tests...")
    
    try:
        # Test imports
        import cv2
        import pytesseract
        import easyocr
        from PIL import Image
        import pdf2image
        import pillow_heif
        print("✅ All imports successful")
        
        # Test OCR
        import numpy as np
        test_image = np.zeros((100, 300, 3), dtype=np.uint8)
        test_image.fill(255)  # White background
        
        # Create a simple test with EasyOCR
        reader = easyocr.Reader(['en'])
        print("✅ EasyOCR initialized successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🗺️  ImagetoGeoCoordinates Setup")
    print("=" * 40)
    
    # Create directories
    create_directories()
    
    # Check system dependencies
    if not check_system_dependencies():
        print("\n❌ Please install missing system dependencies before continuing")
        sys.exit(1)
    
    # Install Python requirements
    if not install_requirements():
        print("\n❌ Failed to install Python dependencies")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("\n❌ Tests failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo run the application:")
    print("  python app.py")
    print("\nThen open: http://localhost:5000")

if __name__ == "__main__":
    main() 