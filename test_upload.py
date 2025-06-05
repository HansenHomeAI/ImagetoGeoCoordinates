#!/usr/bin/env python3

import requests
import json
import sys

def test_parcel_processing():
    """Test the parcel map processing with the provided PDF"""
    
    url = "http://localhost:8080/upload"
    pdf_file = "LOT 2 324 Dolan Rd Aerial Map.pdf"
    
    print(f"🚀 Testing parcel map processing with: {pdf_file}")
    
    try:
        with open(pdf_file, 'rb') as f:
            files = {'file': (pdf_file, f, 'application/pdf')}
            
            print("📤 Uploading file to processing server...")
            response = requests.post(url, files=files, timeout=120)
            
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Processing successful!")
                print("\n📄 RESULTS:")
                print("=" * 50)
                print(json.dumps(result, indent=2))
                
                # Summary
                print("\n📊 SUMMARY:")
                print(f"✅ Success: {result.get('success', False)}")
                print(f"📝 Text length: {len(result.get('extracted_text', ''))}")
                print(f"🏠 Shapes detected: {result.get('detected_shapes', 0)}")
                print(f"📍 Property coordinates: {len(result.get('property_coordinates', []))}")
                print(f"🌍 Location found: {'Yes' if result.get('base_coordinates') else 'No'}")
                
                if result.get('extracted_text'):
                    print(f"\n📄 EXTRACTED TEXT (first 500 chars):")
                    print("-" * 50)
                    print(result['extracted_text'][:500])
                    
            else:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
                
    except FileNotFoundError:
        print(f"❌ Error: File '{pdf_file}' not found")
        return False
    except requests.ConnectionError:
        print("❌ Error: Could not connect to server. Is it running on port 8080?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Parcel Map Processing Test")
    print("=" * 40)
    
    # Test health endpoint first
    try:
        health_response = requests.get("http://localhost:8080/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Server is healthy")
        else:
            print("⚠️ Server health check failed")
    except:
        print("❌ Server is not responding")
        sys.exit(1)
    
    # Run the test
    success = test_parcel_processing()
    sys.exit(0 if success else 1) 