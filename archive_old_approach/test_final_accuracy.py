#!/usr/bin/env python3

import json
from geopy.distance import geodesic

# Load the fixed results
with open('lot2_final_fixed_results.json', 'r') as f:
    results = json.load(f)

print('=== FINAL COORDINATE ACCURACY TEST ===')

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
    print(f"Original Vertex: {first_vertex['original_lat']:.6f}, {first_vertex['original_lon']:.6f}")
    
    # Calculate distance from base to first vertex
    base_point = (base_coords['latitude'], base_coords['longitude'])
    first_point = (first_vertex['latitude'], first_vertex['longitude'])
    original_point = (first_vertex['original_lat'], first_vertex['original_lon'])
    
    distance = geodesic(base_point, first_point).meters
    original_distance = geodesic(base_point, original_point).meters
    
    print(f"\nDistance from base (original): {original_distance:.1f} meters")
    print(f"Distance from base (fixed): {distance:.1f} meters")
    print(f"Improvement: {original_distance - distance:.1f} meters closer")
    
    # Check if coordinates are in Washington State
    wa_bounds = {'min_lat': 45.543, 'max_lat': 49.002, 'min_lon': -124.844, 'max_lon': -116.915}
    
    lat, lon = first_vertex['latitude'], first_vertex['longitude']
    in_wa = (wa_bounds['min_lat'] <= lat <= wa_bounds['max_lat'] and
             wa_bounds['min_lon'] <= lon <= wa_bounds['max_lon'])
    
    print(f"\nFirst vertex in Washington State: {in_wa}")
    
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

print(f"\n=== ACCURACY ASSESSMENT ===")

# Count successes
successes = 0
total_tests = 6

if in_wa:
    print("✅ PASS: Coordinates in Washington State")
    successes += 1
else:
    print("❌ FAIL: Coordinates not in Washington State")

if lat >= 46.0 and lat <= 46.5:
    print("✅ PASS: Coordinates in Cowlitz County area")
    successes += 1
else:
    print("❌ FAIL: Coordinates not in Cowlitz County area")

if distance < 5000:
    print("✅ PASS: Reasonable distance from base location")
    successes += 1
else:
    print("❌ FAIL: Too far from base location")

if 50 <= lat_span_meters <= 500:
    print("✅ PASS: Reasonable property width")
    successes += 1
else:
    print("❌ FAIL: Property width unreasonable")

if 50 <= lon_span_meters <= 500:
    print("✅ PASS: Reasonable property length")
    successes += 1
else:
    print("❌ FAIL: Property length unreasonable")

# Calculate area
area = 0
if len(first_shape) >= 3:
    # Simple area calculation
    total_area = 0
    n = len(first_shape)
    
    for i in range(n):
        j = (i + 1) % n
        lat1, lon1 = first_shape[i]['latitude'], first_shape[i]['longitude']
        lat2, lon2 = first_shape[j]['latitude'], first_shape[j]['longitude']
        
        # Convert to approximate meters
        lat_avg = (lat1 + lat2) / 2
        meters_per_deg_lat = 111000
        meters_per_deg_lon = 111000 * 0.6  # Rough adjustment
        
        x1 = lon1 * meters_per_deg_lon
        y1 = lat1 * meters_per_deg_lat
        x2 = lon2 * meters_per_deg_lon
        y2 = lat2 * meters_per_deg_lat
        
        total_area += (x1 * y2 - x2 * y1)
    
    area = abs(total_area) / 2

if 100 <= area <= 5000:
    print("✅ PASS: Reasonable property area")
    successes += 1
else:
    print("❌ FAIL: Property area unreasonable")

print(f"\n🏆 OVERALL SCORE: {successes}/{total_tests} ({successes/total_tests*100:.1f}%)")

if successes >= 5:
    print("🎉 EXCELLENT: Coordinates are highly accurate!")
elif successes >= 4:
    print("✅ GOOD: Coordinates are reasonably accurate")
elif successes >= 3:
    print("⚠️ FAIR: Coordinates need some improvement")
else:
    print("❌ POOR: Coordinates need major improvement")

print(f"\nProperty area: {area:.1f} square meters")
print(f"Scale used: {results['fix_details']['scale_info']['scale_meters_per_pixel']:.3f} m/pixel") 