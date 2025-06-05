#!/usr/bin/env python3
"""
Test script for enhanced parcel map processing
"""
import requests
import os
import json

def test_parcel_map_processing():
    """Test the enhanced parcel map processing with actual file"""
    
    # Configuration
    server_url = "http://localhost:8081"
    test_file = "LOT 2 324 Dolan Rd Aerial Map.pdf"
    
    print("🧪 Testing Enhanced Parcel Map Processing")
    print("=" * 50)
    
    # Check if test file exists
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        print("Please ensure the parcel map file is in the current directory")
        return False
    
    print(f"📄 Test file: {test_file}")
    print(f"📏 File size: {os.path.getsize(test_file) / 1024:.1f} KB")
    
    # Check server health
    try:
        health_response = requests.get(f"{server_url}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Server is healthy")
        else:
            print(f"⚠️ Server health check failed: {health_response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    # Upload and process the file
    print("\n🚀 Starting file upload and processing...")
    print("This may take some time due to enhanced processing...")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'application/pdf')}
            
            print("📤 Uploading file...")
            response = requests.post(
                f"{server_url}/upload", 
                files=files, 
                timeout=120  # 2 minutes timeout for processing
            )
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS! Processing completed")
            print("=" * 30)
            
            # Display results
            print(f"📄 Extracted text length: {len(result.get('extracted_text', ''))}")
            print(f"🔍 Text preview: {result.get('extracted_text', '')[:200]}...")
            print(f"\n🏠 Shapes detected: {len(result.get('coordinates', []))}")
            
            # Location information
            location = result.get('location_info', {})
            print(f"\n📍 Location Information:")
            print(f"   Streets: {location.get('streets', [])}")
            print(f"   County: {location.get('county', 'Not found')}")
            print(f"   State: {location.get('state', 'Not found')}")
            print(f"   Coordinates found: {location.get('coordinates', [])}")
            
            # Base coordinates
            base_coords = result.get('base_coordinates')
            if base_coords:
                print(f"\n🌐 Base Coordinates:")
                print(f"   Latitude: {base_coords.get('latitude')}")
                print(f"   Longitude: {base_coords.get('longitude')}")
                print(f"   Address: {base_coords.get('address', 'N/A')}")
            
            # Shape coordinates
            coordinates = result.get('coordinates', [])
            for i, shape in enumerate(coordinates):
                print(f"\n🏠 Shape {i+1} ({len(shape)} vertices):")
                for j, coord in enumerate(shape[:3]):  # Show first 3 vertices
                    print(f"   Vertex {j+1}: ({coord['latitude']:.6f}, {coord['longitude']:.6f})")
                if len(shape) > 3:
                    print(f"   ... and {len(shape) - 3} more vertices")
            
            return True
            
        else:
            print(f"❌ Processing failed with status {response.status_code}")
            try:
                error_info = response.json()
                print(f"Error: {error_info.get('error', 'Unknown error')}")
            except:
                print(f"Error response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - processing may be taking longer than expected")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_parcel_map_processing()
    exit(0 if success else 1) 