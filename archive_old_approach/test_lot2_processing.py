#!/usr/bin/env python3
"""
Test script to process LOT 2 324 Dolan Rd Aerial Map.pdf and validate results
"""

import requests
import json
import time
import sys

def test_lot2_processing():
    """Test processing of LOT 2 file with comprehensive validation"""
    
    print("🚀 Starting LOT 2 324 Dolan Rd processing test...")
    
    # File to test
    file_path = "LOT 2 324 Dolan Rd Aerial Map.pdf"
    url = "http://localhost:8081/upload"
    
    try:
        # Upload and process the file
        print(f"📤 Uploading file: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'application/pdf')}
            response = requests.post(url, files=files, timeout=300)  # 5 minute timeout
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Processing completed successfully!")
            print(f"📊 Response structure: {list(result.keys())}")
            
            # Analyze results
            analyze_results(result)
            
            return result
        else:
            print(f"❌ Processing failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def analyze_results(result):
    """Analyze the processing results in detail"""
    
    print("\n" + "="*60)
    print("📊 DETAILED RESULT ANALYSIS")
    print("="*60)
    
    # Check if processing was successful
    if not result.get('success', False):
        print("❌ Processing failed!")
        if 'error' in result:
            print(f"Error: {result['error']}")
        return
    
    # Text extraction analysis
    extracted_text = result.get('extracted_text', '')
    print(f"\n📝 TEXT EXTRACTION:")
    print(f"   Length: {len(extracted_text)} characters")
    if extracted_text:
        print(f"   Preview: {extracted_text[:200]}...")
        
        # Look for key information
        if '324' in extracted_text and 'dolan' in extracted_text.lower():
            print("   ✅ Address information found")
        else:
            print("   ⚠️ Address information not clearly found")
    
    # Location analysis
    location_info = result.get('location_info', {})
    print(f"\n🌍 LOCATION INFORMATION:")
    print(f"   Streets found: {len(location_info.get('streets', []))}")
    print(f"   County: {location_info.get('county', 'Not found')}")
    print(f"   State: {location_info.get('state', 'Not found')}")
    print(f"   Addresses: {location_info.get('addresses', [])}")
    
    # Base coordinates analysis
    base_coords = result.get('base_coordinates', {})
    print(f"\n📍 BASE COORDINATES:")
    if base_coords:
        lat = base_coords.get('latitude', 0)
        lon = base_coords.get('longitude', 0)
        print(f"   Latitude: {lat}")
        print(f"   Longitude: {lon}")
        print(f"   Address: {base_coords.get('address', 'Unknown')}")
        
        # Check if coordinates are in Washington State
        if 45.5 <= lat <= 49.0 and -125.0 <= lon <= -116.9:
            print("   ✅ Coordinates are in Washington State")
        else:
            print("   ❌ Coordinates are NOT in Washington State")
        
        # Check if coordinates are in Cowlitz County area
        cowlitz_center = (46.096, -122.621)
        distance_from_center = ((lat - cowlitz_center[0])**2 + (lon - cowlitz_center[1])**2)**0.5
        if distance_from_center < 0.5:  # Within 0.5 degrees
            print("   ✅ Coordinates are near Cowlitz County")
        else:
            print(f"   ⚠️ Coordinates are {distance_from_center:.2f} degrees from Cowlitz County center")
    else:
        print("   ❌ No base coordinates found")
    
    # Property coordinates analysis
    property_coords = result.get('property_coordinates', [])
    print(f"\n🏠 PROPERTY COORDINATES:")
    print(f"   Number of shapes: {len(property_coords)}")
    
    total_vertices = 0
    for i, shape in enumerate(property_coords):
        if isinstance(shape, list):
            num_coords = len(shape)
        elif isinstance(shape, dict):
            num_coords = len(shape.get('coordinates', []))
        else:
            num_coords = 0
        
        total_vertices += num_coords
        print(f"   Shape {i+1}: {num_coords} vertices")
        
        # Analyze first few coordinates of first shape
        if i == 0 and num_coords > 0:
            coords_to_analyze = shape if isinstance(shape, list) else shape.get('coordinates', [])
            if coords_to_analyze:
                first_coord = coords_to_analyze[0]
                if isinstance(first_coord, dict):
                    coord_lat = first_coord.get('latitude', 0)
                    coord_lon = first_coord.get('longitude', 0)
                    print(f"   First vertex: ({coord_lat}, {coord_lon})")
                    
                    # Check coordinate reasonableness
                    if 45.5 <= coord_lat <= 49.0 and -125.0 <= coord_lon <= -116.9:
                        print("   ✅ First vertex in Washington State")
                    else:
                        print("   ❌ First vertex NOT in Washington State")
    
    print(f"   Total vertices: {total_vertices}")
    
    # Processing summary
    processing_summary = result.get('processing_summary', {})
    print(f"\n📈 PROCESSING SUMMARY:")
    print(f"   Total shapes: {processing_summary.get('total_shapes', 0)}")
    print(f"   Total coordinates: {processing_summary.get('total_coordinates', 0)}")
    print(f"   Validation score: {processing_summary.get('validation_score', 0):.2f}")
    print(f"   Accuracy score: {processing_summary.get('accuracy_score', 0):.2f}")
    print(f"   Accuracy grade: {processing_summary.get('accuracy_grade', 'Unknown')}")
    print(f"   Corrections applied: {processing_summary.get('corrections_applied', False)}")
    
    # Success assessment
    print(f"\n🎯 SUCCESS ASSESSMENT:")
    
    success_criteria = []
    
    # Check if we have meaningful coordinates
    if total_vertices > 0:
        success_criteria.append("✅ Generated property coordinates")
    else:
        success_criteria.append("❌ No property coordinates generated")
    
    # Check if location is correct
    if base_coords and 45.5 <= base_coords.get('latitude', 0) <= 49.0:
        success_criteria.append("✅ Location in Washington State")
    else:
        success_criteria.append("❌ Location not in Washington State")
    
    # Check if accuracy is reasonable
    accuracy_score = processing_summary.get('accuracy_score', 0)
    if accuracy_score >= 0.7:
        success_criteria.append("✅ High accuracy score")
    elif accuracy_score >= 0.5:
        success_criteria.append("⚠️ Moderate accuracy score")
    else:
        success_criteria.append("❌ Low accuracy score")
    
    # Check if shapes were detected
    detected_shapes = result.get('detected_shapes', 0)
    if detected_shapes > 0:
        success_criteria.append("✅ Property shapes detected")
    else:
        success_criteria.append("❌ No property shapes detected")
    
    for criterion in success_criteria:
        print(f"   {criterion}")
    
    # Overall assessment
    passed_criteria = sum(1 for c in success_criteria if c.startswith("✅"))
    total_criteria = len(success_criteria)
    
    print(f"\n🏆 OVERALL RESULT: {passed_criteria}/{total_criteria} criteria passed")
    
    if passed_criteria >= 3:
        print("🎉 PROCESSING SUCCESSFUL!")
    elif passed_criteria >= 2:
        print("⚠️ PROCESSING PARTIALLY SUCCESSFUL - needs improvement")
    else:
        print("❌ PROCESSING FAILED - significant issues detected")

if __name__ == "__main__":
    print("Starting LOT 2 324 Dolan Rd processing test...")
    result = test_lot2_processing()
    
    if result:
        # Save results for further analysis
        with open('lot2_test_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to lot2_test_results.json")
    else:
        print("\n❌ Test failed - no results to save")
        sys.exit(1) 