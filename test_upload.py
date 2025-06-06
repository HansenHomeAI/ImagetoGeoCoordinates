#!/usr/bin/env python3
"""
Comprehensive test script to upload and analyze parcel maps programmatically.
This will help us identify issues and iterate on improvements.
"""

import requests
import json
import time
import os
from pathlib import Path

def test_parcel_upload(file_path, server_url="http://127.0.0.1:8081"):
    """Test uploading and processing a parcel map file."""
    
    print(f"🧪 Testing parcel map upload: {file_path}")
    print(f"🌐 Server URL: {server_url}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    print(f"📁 File size: {file_size:.2f} MB")
    
    # Test server connectivity first
    try:
        response = requests.get(server_url, timeout=10)
        if response.status_code == 200:
            print("✅ Server is responding")
        else:
            print(f"⚠️ Server returned status: {response.status_code}")
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        return None
    
    # Upload and process the file
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
            
            print("📤 Uploading file...")
            start_time = time.time()
            
            response = requests.post(
                f"{server_url}/upload",
                files=files,
                timeout=300  # 5 minute timeout for processing
            )
            
            processing_time = time.time() - start_time
            print(f"⏱️ Processing time: {processing_time:.2f} seconds")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Upload successful!")
                return analyze_results(result)
            else:
                print(f"❌ Upload failed with status: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
    except requests.exceptions.Timeout:
        print("❌ Upload timed out (processing took too long)")
        return None
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None

def analyze_results(result):
    """Analyze the processing results and identify areas for improvement."""
    
    print("\n" + "="*60)
    print("📊 ANALYSIS RESULTS")
    print("="*60)
    
    # Check for errors
    if result.get('error'):
        print(f"❌ Processing Error: {result['error']}")
        return result
    
    # Analyze text extraction
    text_data = result.get('text_extraction', {})
    ocr_text = text_data.get('raw_text', '')
    street_names = text_data.get('street_names', [])
    coordinates = text_data.get('coordinates', [])
    counties = text_data.get('counties', [])
    states = text_data.get('states', [])
    
    print(f"📝 Raw Text Length: {len(ocr_text)} characters")
    print(f"🛣️ Street Names Found: {len(street_names)} - {street_names}")
    print(f"📍 Coordinates Found: {len(coordinates)} - {coordinates}")
    print(f"🏛️ Counties Found: {len(counties)} - {counties}")
    print(f"🗺️ States Found: {len(states)} - {states}")
    
    # Analyze shape detection
    shape_data = result.get('shape_detection', {})
    property_boundaries = shape_data.get('property_boundaries', [])
    other_shapes = shape_data.get('other_shapes', [])
    
    print(f"🏠 Property Boundaries: {len(property_boundaries)}")
    print(f"🔷 Other Shapes: {len(other_shapes)}")
    
    # Analyze geocoding
    geocoding_data = result.get('geocoding', {})
    base_location = geocoding_data.get('base_location')
    
    if base_location:
        print(f"🌍 Base Location: {base_location.get('address', 'Unknown')}")
        print(f"🎯 Base Coordinates: {base_location.get('lat')}, {base_location.get('lon')}")
    else:
        print("❌ No base location found")
    
    # Analyze final coordinates
    final_coords = result.get('coordinates', [])
    print(f"🎯 Final Coordinates Generated: {len(final_coords)}")
    
    for i, coord in enumerate(final_coords[:5]):  # Show first 5
        print(f"   Point {i+1}: {coord.get('lat'):.6f}, {coord.get('lon'):.6f}")
    
    # Identify issues and improvement areas
    print("\n" + "="*60)
    print("🔍 IMPROVEMENT ANALYSIS")
    print("="*60)
    
    issues = []
    recommendations = []
    
    if len(ocr_text) < 100:
        issues.append("Very little text extracted")
        recommendations.append("Improve OCR preprocessing and multi-engine approach")
    
    if len(street_names) == 0:
        issues.append("No street names found")
        recommendations.append("Enhance street name regex patterns and OCR quality")
    
    if len(coordinates) == 0:
        issues.append("No coordinate references found")
        recommendations.append("Add support for more coordinate formats (UTM, State Plane, etc.)")
    
    if not base_location:
        issues.append("Failed to geocode location")
        recommendations.append("Implement fallback geocoding strategies")
    
    if len(property_boundaries) == 0:
        issues.append("No property boundaries detected")
        recommendations.append("Improve edge detection and contour filtering")
    
    if len(final_coords) == 0:
        issues.append("No final coordinates generated")
        recommendations.append("Implement coordinate estimation algorithms")
    
    print("❌ Issues Found:")
    for issue in issues:
        print(f"   • {issue}")
    
    print("\n💡 Recommendations:")
    for rec in recommendations:
        print(f"   • {rec}")
    
    return result

def main():
    """Main test function."""
    
    # Look for parcel map files
    possible_files = [
        "LOT 2 324 Dolan Rd Aerial Map.pdf",
        "parcel_map.pdf",
        "test_parcel.pdf"
    ]
    
    parcel_file = None
    for filename in possible_files:
        if os.path.exists(filename):
            parcel_file = filename
            break
    
    if not parcel_file:
        print("❌ No parcel map file found. Please ensure you have:")
        for filename in possible_files:
            print(f"   • {filename}")
        return
    
    # Test the upload
    result = test_parcel_upload(parcel_file)
    
    if result:
        # Save results for further analysis
        with open('test_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to test_results.json")

if __name__ == "__main__":
    main() 