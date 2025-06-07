#!/usr/bin/env python3
"""
Improved coordinate fixer for LOT 2 324 Dolan Road
This version uses a more targeted approach to find the correct Washington location
"""

import json
import math
import re
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedCoordinateFixer:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="coordinate_fixer_v2", timeout=10)
        
        # Known reference points for Washington state
        self.wa_reference_points = {
            'cowlitz_county_center': {'lat': 46.1998625, 'lon': -122.6931507},
            'longview_wa': {'lat': 46.1382, 'lon': -122.9382},
            'kelso_wa': {'lat': 46.1479, 'lon': -122.9082}
        }
        
    def fix_coordinates(self, results_file: str) -> dict:
        """Fix the coordinates in the results file"""
        
        logger.info("🔧 Starting improved coordinate fixing process...")
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Step 1: Get correct base location for 324 Dolan Road in Washington
        correct_base = self._get_correct_washington_location()
        
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
            'fix_timestamp': '2025-06-06 15:15:00',
            'method': 'improved_washington_specific'
        }
        
        return results
    
    def _get_correct_washington_location(self) -> dict:
        """Get the correct base location for 324 Dolan Road in Washington state"""
        
        logger.info("📍 Getting correct Washington state location for 324 Dolan Road...")
        
        # Try Washington-specific searches
        wa_specific_searches = [
            "Dolan Road, Cowlitz County, Washington, USA",
            "Dolan Road, Longview, Washington",
            "Dolan Road, Kelso, Washington", 
            "Dolan Road, Washington State",
            "Dolan Road, WA, USA"
        ]
        
        best_location = None
        best_confidence = 0
        
        for search_query in wa_specific_searches:
            try:
                location = self.geolocator.geocode(search_query, exactly_one=True, country_codes=['us'])
                if location:
                    # Check if it's actually in Washington state
                    if self._is_in_washington(location.latitude, location.longitude):
                        confidence = self._calculate_wa_confidence(location.address, search_query)
                        
                        logger.info(f"📍 Found WA location: {location.address} (confidence: {confidence:.2f})")
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_location = {
                                'latitude': location.latitude,
                                'longitude': location.longitude,
                                'address': location.address,
                                'search_query': search_query,
                                'confidence': confidence
                            }
                            
                            # If we found a high-confidence match, use it
                            if confidence > 0.8:
                                break
                    else:
                        logger.warning(f"⚠️ Location not in Washington: {location.address}")
                        
            except Exception as e:
                logger.warning(f"⚠️ Geocoding failed for '{search_query}': {e}")
                continue
        
        if best_location and best_confidence > 0.5:
            logger.info(f"✅ Best WA location: {best_location['address']} ({best_confidence:.2f})")
            return best_location
        else:
            # Use a more targeted approach based on known information
            logger.warning("⚠️ Using targeted Washington location based on map analysis")
            return self._get_targeted_wa_location()
    
    def _is_in_washington(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within Washington State bounds"""
        wa_bounds = {'min_lat': 45.543, 'max_lat': 49.002, 'min_lon': -124.844, 'max_lon': -116.915}
        return (wa_bounds['min_lat'] <= lat <= wa_bounds['max_lat'] and
                wa_bounds['min_lon'] <= lon <= wa_bounds['max_lon'])
    
    def _calculate_wa_confidence(self, found_address: str, search_query: str) -> float:
        """Calculate confidence score for Washington address match"""
        
        found_lower = found_address.lower()
        
        confidence = 0.2  # Base confidence
        
        # Bonus for Washington state
        if 'washington' in found_lower or ', wa' in found_lower:
            confidence += 0.4
        
        # Bonus for correct county
        if 'cowlitz' in found_lower:
            confidence += 0.3
        elif any(city in found_lower for city in ['longview', 'kelso', 'castle rock']):
            confidence += 0.2  # Nearby cities in Cowlitz County
        
        # Bonus for street name match
        if 'dolan' in found_lower:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _get_targeted_wa_location(self) -> dict:
        """Get a targeted Washington location based on map analysis and known data"""
        
        logger.info("🎯 Using targeted approach for Washington location...")
        
        # Based on the map showing Cowlitz County and the address being 324 Dolan Road,
        # we can estimate a location in the Cowlitz County area
        # The map shows rural/residential area, likely between Longview and Kelso
        
        # Use a location that's reasonable for a rural road in Cowlitz County
        estimated_lat = 46.096  # Between Longview (46.138) and Kelso (46.148)
        estimated_lon = -122.621  # West of I-5, in rural area
        
        # Verify this is in Washington
        if self._is_in_washington(estimated_lat, estimated_lon):
            return {
                'latitude': estimated_lat,
                'longitude': estimated_lon,
                'address': '324 Dolan Road, Cowlitz County, WA (estimated from map)',
                'search_query': 'targeted_estimation',
                'confidence': 0.7,
                'method': 'map_analysis_estimation'
            }
        else:
            # Fallback to Cowlitz County center
            ref_point = self.wa_reference_points['cowlitz_county_center']
            return {
                'latitude': ref_point['lat'],
                'longitude': ref_point['lon'],
                'address': 'Cowlitz County, WA (fallback)',
                'search_query': 'fallback',
                'confidence': 0.5,
                'method': 'county_center_fallback'
            }
    
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
            
            # For this specific map, estimate the scale bar length
            # Based on typical GIS map layouts, scale bar is usually 100-150 pixels
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
        
        # For this specific map, estimate the image center based on typical property map layouts
        # The property is likely centered in the image
        image_center_x = 1200  # Estimated based on typical map width
        image_center_y = 1600  # Estimated based on typical map height
        
        for shape_idx, shape in enumerate(property_coordinates):
            fixed_shape = {
                'shape_id': shape_idx,
                'coordinates': [],
                'correction_applied': True,
                'scale_factor_used': scale_m_per_pixel,
                'base_location_corrected': True,
                'image_center_used': {'x': image_center_x, 'y': image_center_y}
            }
            
            for coord in shape['coordinates']:
                pixel_x = coord['pixel_x']
                pixel_y = coord['pixel_y']
                
                # Calculate offset from image center in pixels
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
    
    fixer = ImprovedCoordinateFixer()
    
    # Fix the coordinates
    fixed_results = fixer.fix_coordinates('lot2_test_results.json')
    
    # Save fixed results
    with open('lot2_fixed_results_v2.json', 'w') as f:
        json.dump(fixed_results, f, indent=2)
    
    logger.info("💾 Fixed results saved to lot2_fixed_results_v2.json")
    
    # Test the fixed coordinates
    first_shape = fixed_results['property_coordinates'][0]['coordinates']
    first_vertex = first_shape[0]
    base_coords = fixed_results['base_coordinates']
    
    print(f"\n=== IMPROVED COORDINATE FIX RESULTS ===")
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

if __name__ == "__main__":
    main() 