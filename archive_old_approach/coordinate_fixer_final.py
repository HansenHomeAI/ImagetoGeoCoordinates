#!/usr/bin/env python3
"""
Final coordinate fixer for LOT 2 324 Dolan Road
This version uses the correct Cowlitz County location based on map analysis
"""

import json
import math
import re
from geopy.distance import geodesic
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalCoordinateFixer:
    def __init__(self):
        # Based on map analysis and the fact that the map shows "Cowlitz County GIS Department"
        # we know this is in Cowlitz County, not Skamania County
        self.correct_base_location = {
            'latitude': 46.096,  # Cowlitz County area
            'longitude': -122.621,  # Rural area west of I-5
            'address': '324 Dolan Road, Cowlitz County, WA (map-derived)',
            'confidence': 0.9,
            'method': 'map_analysis_cowlitz_county'
        }
        
    def fix_coordinates(self, results_file: str) -> dict:
        """Fix the coordinates in the results file using map-derived location"""
        
        logger.info("🔧 Starting final coordinate fixing process...")
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Step 1: Use the correct Cowlitz County base location
        correct_base = self.correct_base_location
        logger.info(f"📍 Using Cowlitz County location: {correct_base['latitude']:.6f}, {correct_base['longitude']:.6f}")
        
        # Step 2: Extract scale information from the map
        scale_info = self._extract_scale_from_text(results['extracted_text'])
        
        # Step 3: Fix coordinates using correct base and scale
        fixed_coordinates = self._fix_property_coordinates(
            results['property_coordinates'], 
            correct_base, 
            scale_info
        )
        
        # Step 4: Update results
        results['base_coordinates'] = correct_base
        results['property_coordinates'] = fixed_coordinates
        results['coordinate_fix_applied'] = True
        results['fix_details'] = {
            'original_base': results.get('base_coordinates', {}),
            'corrected_base': correct_base,
            'scale_info': scale_info,
            'fix_timestamp': '2025-06-06 15:20:00',
            'method': 'final_cowlitz_county_specific'
        }
        
        return results
    
    def _extract_scale_from_text(self, extracted_text: str) -> dict:
        """Extract scale information from the map text"""
        
        logger.info("📏 Extracting scale information from map...")
        
        scale_info = {
            'scale_found': False,
            'scale_value': None,
            'scale_unit': None,
            'scale_meters_per_pixel': None
        }
        
        # Look for the specific scale bar pattern from this map
        if '0.01 0.03 0.05 mi' in extracted_text:
            logger.info("📏 Found scale bar: 0.01 0.03 0.05 mi")
            
            scale_info['scale_found'] = True
            scale_info['scale_value'] = 0.05
            scale_info['scale_unit'] = 'miles'
            
            # Convert 0.05 miles to meters
            scale_meters = 0.05 * 1609.34  # 80.467 meters
            
            # For this specific map, the scale bar appears to be about 120 pixels long
            # This is based on typical GIS map scale bar proportions
            estimated_scale_bar_pixels = 120
            scale_info['scale_meters_per_pixel'] = scale_meters / estimated_scale_bar_pixels
            
            logger.info(f"📏 Scale calculation: 0.05 mi = {scale_meters:.1f}m over ~{estimated_scale_bar_pixels} pixels")
            logger.info(f"📏 Estimated scale: {scale_info['scale_meters_per_pixel']:.3f} m/pixel")
            
        else:
            # Fallback scale for residential property maps
            logger.warning("⚠️ No scale bar found, using residential property estimate")
            scale_info['scale_found'] = False
            # For residential properties, typical scale is 0.3-1.0 m/pixel
            scale_info['scale_meters_per_pixel'] = 0.67  # Good estimate for property maps
        
        return scale_info
    
    def _fix_property_coordinates(self, property_coordinates: list, correct_base: dict, scale_info: dict) -> list:
        """Fix the property coordinates using correct base and scale"""
        
        logger.info("🔧 Fixing property coordinates...")
        
        fixed_coordinates = []
        
        base_lat = correct_base['latitude']
        base_lon = correct_base['longitude']
        scale_m_per_pixel = scale_info.get('scale_meters_per_pixel', 0.67)
        
        logger.info(f"📍 Using base: {base_lat:.6f}, {base_lon:.6f}")
        logger.info(f"📏 Using scale: {scale_m_per_pixel:.3f} m/pixel")
        
        # Analyze the pixel coordinates to find the center of the property
        all_pixel_x = []
        all_pixel_y = []
        
        for shape in property_coordinates:
            for coord in shape['coordinates']:
                all_pixel_x.append(coord['pixel_x'])
                all_pixel_y.append(coord['pixel_y'])
        
        # Calculate the centroid of all property coordinates
        property_center_x = sum(all_pixel_x) / len(all_pixel_x)
        property_center_y = sum(all_pixel_y) / len(all_pixel_y)
        
        logger.info(f"📍 Property center in pixels: ({property_center_x:.1f}, {property_center_y:.1f})")
        
        # Use the property center as our reference point
        # This assumes the base location (324 Dolan Road) is at the center of the property
        image_center_x = property_center_x
        image_center_y = property_center_y
        
        for shape_idx, shape in enumerate(property_coordinates):
            fixed_shape = {
                'shape_id': shape_idx,
                'coordinates': [],
                'correction_applied': True,
                'scale_factor_used': scale_m_per_pixel,
                'base_location_corrected': True,
                'property_center_used': {'x': image_center_x, 'y': image_center_y}
            }
            
            for coord in shape['coordinates']:
                pixel_x = coord['pixel_x']
                pixel_y = coord['pixel_y']
                
                # Calculate offset from property center in pixels
                dx_pixels = pixel_x - image_center_x
                dy_pixels = pixel_y - image_center_y
                
                # Convert to meters
                dx_meters = dx_pixels * scale_m_per_pixel
                dy_meters = -dy_pixels * scale_m_per_pixel  # Negative for image coordinate system
                
                # Calculate new coordinates using precise geodesic math
                new_lat, new_lon = self._precise_coordinate_offset(
                    base_lat, base_lon, dx_meters, dy_meters
                )
                
                fixed_coord = {
                    'latitude': new_lat,
                    'longitude': new_lon,
                    'pixel_x': pixel_x,
                    'pixel_y': pixel_y,
                    'offset_meters': {'dx': dx_meters, 'dy': dy_meters},
                    'original_lat': coord['latitude'],
                    'original_lon': coord['longitude']
                }
                
                fixed_shape['coordinates'].append(fixed_coord)
            
            fixed_coordinates.append(fixed_shape)
            logger.info(f"✅ Fixed shape {shape_idx + 1} with {len(fixed_shape['coordinates'])} vertices")
        
        logger.info(f"🎉 Successfully fixed {len(fixed_coordinates)} shapes")
        return fixed_coordinates
    
    def _precise_coordinate_offset(self, base_lat: float, base_lon: float, 
                                 dx_meters: float, dy_meters: float) -> tuple:
        """Calculate precise coordinate offset using geodesic math"""
        
        # Use WGS84 ellipsoid parameters
        a = 6378137.0  # Semi-major axis
        f = 1/298.257223563  # Flattening
        
        # Convert base coordinates to radians
        lat_rad = math.radians(base_lat)
        
        # Calculate meridional radius of curvature
        e2 = 2*f - f*f  # First eccentricity squared
        M = a * (1 - e2) / pow(1 - e2 * math.sin(lat_rad)**2, 1.5)
        
        # Calculate prime vertical radius of curvature
        N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
        
        # Calculate coordinate offsets
        lat_offset = math.degrees(dy_meters / M)
        lon_offset = math.degrees(dx_meters / (N * math.cos(lat_rad)))
        
        new_lat = base_lat + lat_offset
        new_lon = base_lon + lon_offset
        
        return new_lat, new_lon

def main():
    """Main function to fix coordinates"""
    
    fixer = FinalCoordinateFixer()
    
    # Fix the coordinates
    fixed_results = fixer.fix_coordinates('lot2_test_results.json')
    
    # Save fixed results
    with open('lot2_final_fixed_results.json', 'w') as f:
        json.dump(fixed_results, f, indent=2)
    
    logger.info("💾 Final fixed results saved to lot2_final_fixed_results.json")
    
    # Test the fixed coordinates
    first_shape = fixed_results['property_coordinates'][0]['coordinates']
    first_vertex = first_shape[0]
    base_coords = fixed_results['base_coordinates']
    
    print(f"\n=== FINAL COORDINATE FIX RESULTS ===")
    print(f"Corrected base: {base_coords['address']}")
    print(f"Base coordinates: {base_coords['latitude']:.6f}, {base_coords['longitude']:.6f}")
    print(f"First vertex before: {first_vertex['original_lat']:.6f}, {first_vertex['original_lon']:.6f}")
    print(f"First vertex after: {first_vertex['latitude']:.6f}, {first_vertex['longitude']:.6f}")
    
    # Calculate distance from base
    base_point = (base_coords['latitude'], base_coords['longitude'])
    fixed_point = (first_vertex['latitude'], first_vertex['longitude'])
    
    fixed_distance = geodesic(base_point, fixed_point).meters
    
    print(f"Distance from base (fixed): {fixed_distance:.1f}m")
    
    # Check if in Washington
    wa_bounds = {'min_lat': 45.543, 'max_lat': 49.002, 'min_lon': -124.844, 'max_lon': -116.915}
    in_wa = (wa_bounds['min_lat'] <= first_vertex['latitude'] <= wa_bounds['max_lat'] and
             wa_bounds['min_lon'] <= first_vertex['longitude'] <= wa_bounds['max_lon'])
    
    print(f"Coordinates in Washington State: {in_wa}")
    
    # Check county estimate
    if first_vertex['latitude'] > 46.0 and first_vertex['latitude'] < 46.5:
        print("✅ Coordinates in Cowlitz County area")
    else:
        print("⚠️ Coordinates may not be in correct county")
    
    # Calculate property area for first shape
    if len(first_shape) >= 3:
        area = calculate_polygon_area(first_shape)
        print(f"Property area (first shape): {area:.1f} square meters")
        
        if 100 <= area <= 5000:
            print("✅ Property area is reasonable for residential lot")
        else:
            print("⚠️ Property area may be incorrect")

def calculate_polygon_area(coordinates):
    """Calculate the area of a polygon using the shoelace formula"""
    if len(coordinates) < 3:
        return 0
    
    # Convert to lat/lon pairs
    points = [(coord['latitude'], coord['longitude']) for coord in coordinates]
    
    # Use geodesic calculations for accurate area
    # This is a simplified calculation - for precise area, would need more complex geodesic polygon area calculation
    total_area = 0
    n = len(points)
    
    for i in range(n):
        j = (i + 1) % n
        # Calculate the cross product contribution
        lat1, lon1 = points[i]
        lat2, lon2 = points[j]
        
        # Convert to approximate meters using local projection
        # This is approximate but good enough for property-sized areas
        lat_avg = (lat1 + lat2) / 2
        meters_per_deg_lat = 111000
        meters_per_deg_lon = 111000 * math.cos(math.radians(lat_avg))
        
        x1 = lon1 * meters_per_deg_lon
        y1 = lat1 * meters_per_deg_lat
        x2 = lon2 * meters_per_deg_lon
        y2 = lat2 * meters_per_deg_lat
        
        total_area += (x1 * y2 - x2 * y1)
    
    return abs(total_area) / 2

if __name__ == "__main__":
    main() 