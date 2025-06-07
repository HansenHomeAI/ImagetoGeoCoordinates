#!/usr/bin/env python3

import json
from geopy.distance import geodesic

# Load the results
with open('lot2_test_results.json', 'r') as f:
    results = json.load(f)

print('=== COORDINATE ANALYSIS ===')

# Base coordinates
base_coords = results['base_coordinates']
print(f"Base Location: {base_coords['latitude']:.6f}, {base_coords['longitude']:.6f}")
print(f"Base Address: {base_coords['address']}")

# Property coordinates
property_coords = results['property_coordinates']
print(f"\nProperty Shapes: {len(property_coords)}")

# Analyze first shape
if property_coords:
    first_shape = property_coords[0]['coordinates']
    first_vertex = first_shape[0]
    
    print(f"\nFirst Vertex: {first_vertex['latitude']:.6f}, {first_vertex['longitude']:.6f}")
    
    # Calculate distance from base to first vertex
    base_point = (base_coords['latitude'], base_coords['longitude'])
    first_point = (first_vertex['latitude'], first_vertex['longitude'])
    distance = geodesic(base_point, first_point).meters
    
    print(f"Distance from base to first vertex: {distance:.1f} meters")
    
    # Check if coordinates are in Washington State
    wa_bounds = {'min_lat': 45.543, 'max_lat': 49.002, 'min_lon': -124.844, 'max_lon': -116.915}
    
    lat, lon = first_vertex['latitude'], first_vertex['longitude']
    in_wa = (wa_bounds['min_lat'] <= lat <= wa_bounds['max_lat'] and
             wa_bounds['min_lon'] <= lon <= wa_bounds['max_lon'])
    
    print(f"First vertex in Washington State: {in_wa}")
    
    # Check county (rough estimate)
    if lat < 46.0:
        likely_county = "Skamania County (too far south)"
    elif lat > 46.5:
        likely_county = "Lewis County (too far north)"
    else:
        likely_county = "Cowlitz County area (correct)"
    
    print(f"Likely county: {likely_county}")
    
    # Analyze coordinate spread
    all_lats = [coord['latitude'] for coord in first_shape]
    all_lons = [coord['longitude'] for coord in first_shape]
    
    lat_span = max(all_lats) - min(all_lats)
    lon_span = max(all_lons) - min(all_lons)
    
    # Convert to approximate meters
    lat_span_meters = lat_span * 111000
    lon_span_meters = lon_span * 111000 * 0.6  # Rough cos(latitude) adjustment
    
    print(f"\nProperty span: {lat_span_meters:.1f}m x {lon_span_meters:.1f}m")
    
    # Check if reasonable property size
    if lat_span_meters < 10 or lon_span_meters < 10:
        print("⚠️  Property too small - scale issue")
    elif lat_span_meters > 1000 or lon_span_meters > 1000:
        print("⚠️  Property too large - scale issue")
    else:
        print("✅ Property size reasonable")

print(f"\n=== ISSUES IDENTIFIED ===")
if distance > 1000:
    print(f"❌ MAJOR: Coordinates {distance:.0f}m from base location")
if lat < 46.0:
    print("❌ MAJOR: Coordinates in wrong county (too far south)")
if lat_span_meters < 50:
    print("❌ SCALE: Property too small, scale factor incorrect") 