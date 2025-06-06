#!/usr/bin/env python3
"""
Results Validation Tool - Analyze the quality of detected property boundaries
"""

import requests
import json
import math
from geopy.distance import geodesic

def test_parcel_processing():
    """Test and validate parcel processing results"""
    print("🧪 COMPREHENSIVE PARCEL PROCESSING VALIDATION")
    print("=" * 60)
    
    # Test the parcel map
    url = "http://localhost:8081/upload"
    
    try:
        with open("LOT 2 324 Dolan Rd Aerial Map.pdf", "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ PROCESSING SUCCESS")
            print("-" * 30)
            print(f"📄 Extracted text: {len(data.get('extracted_text', ''))} characters")
            print(f"🏠 Shapes detected: {data.get('detected_shapes', 0)}")
            print(f"📍 Coordinate sets: {len(data.get('property_coordinates', []))}")
            
            # Analyze base coordinates
            base_coords = data.get('base_coordinates', {})
            if base_coords:
                print(f"\n🌐 BASE LOCATION ANALYSIS")
                print("-" * 30)
                print(f"📍 Latitude: {base_coords.get('latitude')}")
                print(f"📍 Longitude: {base_coords.get('longitude')}")
                print(f"🏠 Address: {base_coords.get('address')}")
                
                # Validate coordinates are in Washington state
                lat, lon = base_coords.get('latitude'), base_coords.get('longitude')
                if lat and lon:
                    if 45.5 <= lat <= 49.0 and -124.8 <= lon <= -116.9:
                        print("✅ Coordinates are within Washington state bounds")
                    else:
                        print("⚠️ Coordinates may be outside Washington state")
            
            # Analyze detected properties
            properties = data.get('property_coordinates', [])
            if properties:
                print(f"\n🏠 PROPERTY ANALYSIS")
                print("-" * 30)
                
                for i, prop in enumerate(properties):
                    print(f"\n📐 Property {i+1}:")
                    print(f"   Vertices: {len(prop)}")
                    
                    if len(prop) >= 3:
                        # Calculate approximate area using shoelace formula
                        area_sq_meters = calculate_polygon_area(prop)
                        area_acres = area_sq_meters * 0.000247105  # Convert to acres
                        
                        print(f"   Area: ~{area_acres:.3f} acres ({area_sq_meters:.1f} sq meters)")
                        
                        # Check if it's a reasonable property size
                        if 0.01 <= area_acres <= 100:
                            print("   ✅ Reasonable property size")
                        else:
                            print("   ⚠️ Unusual property size")
                        
                        # Analyze shape characteristics
                        perimeter = calculate_perimeter(prop)
                        compactness = (4 * math.pi * area_sq_meters) / (perimeter ** 2) if perimeter > 0 else 0
                        
                        print(f"   Perimeter: {perimeter:.1f} meters")
                        print(f"   Compactness: {compactness:.3f} (1.0 = perfect circle)")
                        
                        # Show first few coordinates
                        print("   Coordinates (first 3):")
                        for j, coord in enumerate(prop[:3]):
                            print(f"     {j+1}: {coord['latitude']:.6f}, {coord['longitude']:.6f}")
                        if len(prop) > 3:
                            print(f"     ... and {len(prop)-3} more vertices")
            
            # Location information analysis
            location_info = data.get('location_info', {})
            if location_info:
                print(f"\n🔍 LOCATION INFORMATION ANALYSIS")
                print("-" * 30)
                
                streets = location_info.get('streets', [])
                print(f"📍 Streets found: {len(streets)}")
                if streets:
                    print("   Key streets:")
                    for street in streets[:5]:  # Show first 5
                        if len(street) > 10:  # Filter out noise
                            print(f"     - {street}")
                
                county = location_info.get('county')
                state = location_info.get('state')
                if county:
                    print(f"🏛️ County: {county}")
                if state:
                    print(f"🗺️ State: {state}")
                
                lot_numbers = location_info.get('lot_numbers', [])
                if lot_numbers:
                    print(f"🏠 Lot numbers: {lot_numbers}")
            
            # Overall assessment
            print(f"\n🎯 OVERALL ASSESSMENT")
            print("-" * 30)
            
            score = 0
            max_score = 5
            
            if data.get('detected_shapes', 0) > 0:
                score += 1
                print("✅ Property boundaries detected")
            else:
                print("❌ No property boundaries detected")
            
            if len(properties) > 0:
                score += 1
                print("✅ Coordinates generated")
            else:
                print("❌ No coordinates generated")
            
            if base_coords and base_coords.get('latitude'):
                score += 1
                print("✅ Base location identified")
            else:
                print("❌ No base location found")
            
            if location_info.get('state') == 'WA':
                score += 1
                print("✅ Correct state identified")
            else:
                print("❌ State not correctly identified")
            
            if any(0.01 <= calculate_polygon_area(prop) * 0.000247105 <= 100 for prop in properties if len(prop) >= 3):
                score += 1
                print("✅ Reasonable property sizes detected")
            else:
                print("❌ Property sizes may be unrealistic")
            
            print(f"\n📊 Success Score: {score}/{max_score} ({score/max_score*100:.1f}%)")
            
            if score >= 4:
                print("🎉 EXCELLENT - System is working very well!")
            elif score >= 3:
                print("✅ GOOD - System is working with minor issues")
            elif score >= 2:
                print("⚠️ FAIR - System needs improvement")
            else:
                print("❌ POOR - System needs significant work")
            
            return score >= 3
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"💥 Error during testing: {e}")
        return False

def calculate_polygon_area(coordinates):
    """Calculate polygon area using the shoelace formula"""
    if len(coordinates) < 3:
        return 0
    
    # Convert to meters using approximate conversion
    # This is rough but good enough for validation
    coords_meters = []
    for coord in coordinates:
        lat, lon = coord['latitude'], coord['longitude']
        # Convert to approximate meters (very rough)
        x = lon * 111320 * math.cos(math.radians(lat))  # meters
        y = lat * 110540  # meters
        coords_meters.append((x, y))
    
    # Shoelace formula
    area = 0
    n = len(coords_meters)
    for i in range(n):
        j = (i + 1) % n
        area += coords_meters[i][0] * coords_meters[j][1]
        area -= coords_meters[j][0] * coords_meters[i][1]
    
    return abs(area) / 2

def calculate_perimeter(coordinates):
    """Calculate polygon perimeter"""
    if len(coordinates) < 2:
        return 0
    
    perimeter = 0
    for i in range(len(coordinates)):
        j = (i + 1) % len(coordinates)
        coord1 = (coordinates[i]['latitude'], coordinates[i]['longitude'])
        coord2 = (coordinates[j]['latitude'], coordinates[j]['longitude'])
        distance = geodesic(coord1, coord2).meters
        perimeter += distance
    
    return perimeter

if __name__ == "__main__":
    success = test_parcel_processing()
    if success:
        print("\n🚀 Ready for testing with different parcel map formats!")
    else:
        print("\n🔧 System needs refinement before testing other formats.") 