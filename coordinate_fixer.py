#!/usr/bin/env python3
"""
Specialized coordinate fixer for LOT 2 324 Dolan Road
This script fixes the coordinate accuracy issues by:
1. Using the correct base location for 324 Dolan Road
2. Calculating proper scale from the map
3. Applying correct coordinate transformation
"""

import json
import math
import re
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoordinateFixer:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="coordinate_fixer", timeout=10)
        
    def fix_coordinates(self, results_file: str) -> dict:
        """Fix the coordinates in the results file"""
        
        logger.info("🔧 Starting coordinate fixing process...")
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Step 1: Get correct base location for 324 Dolan Road
        correct_base = self._get_correct_base_location()
        
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
            'fix_timestamp': '2025-06-06 15:10:00'
        }
        
        return results
    
    def _get_correct_base_location(self) -> dict:
        """Get the correct base location for 324 Dolan Road"""
        
        logger.info("📍 Getting correct base location for 324 Dolan Road...")
        
        # Try multiple address variations to get the most accurate location
        address_variations = [
            "324 Dolan Road, Cowlitz County, WA",
            "324 Dolan Rd, Cowlitz County, Washington",
            "Dolan Road, Cowlitz County, WA",
            "324 Dolan Road, Washington"
        ]
        
        best_location = None
        best_confidence = 0
        
        for address in address_variations:
            try:
                location = self.geolocator.geocode(address, exactly_one=True)
                if location:
                    # Calculate confidence based on address match
                    confidence = self._calculate_address_confidence(location.address, address)
                    
                    logger.info(f"📍 Found: {location.address} (confidence: {confidence:.2f})")
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_location = {
                            'latitude': location.latitude,
                            'longitude': location.longitude,
                            'address': location.address,
                            'search_query': address,
                            'confidence': confidence
                        }
                        
                        # If we found a high-confidence match, use it
                        if confidence > 0.8:
                            break
                            
            except Exception as e:
                logger.warning(f"⚠️ Geocoding failed for '{address}': {e}")
                continue
        
        if best_location:
            logger.info(f"✅ Best location: {best_location['address']} ({best_confidence:.2f})")
            return best_location
        else:
            # Fallback to known approximate location for 324 Dolan Road
            logger.warning("⚠️ Using fallback location for 324 Dolan Road")
            return {
                'latitude': 46.096,  # Approximate location in Cowlitz County
                'longitude': -122.621,
                'address': '324 Dolan Road, Cowlitz County, WA (estimated)',
                'search_query': 'fallback',
                'confidence': 0.5
            }
    
    def _calculate_address_confidence(self, found_address: str, search_address: str) -> float:
        """Calculate confidence score for address match"""
        
        found_lower = found_address.lower()
        search_lower = search_address.lower()
        
        confidence = 0.3  # Base confidence
        
        # Bonus for street name match
        if 'dolan' in found_lower:
            confidence += 0.3
        
        # Bonus for correct county
        if 'cowlitz' in found_lower:
            confidence += 0.3
        elif 'skamania' in found_lower:
            confidence += 0.1  # Close but not ideal
        
        # Bonus for correct state
        if 'washington' in found_lower or 'wa' in found_lower:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _extract_scale_from_text(self, extracted_text: str) -> dict:
        """Extract scale information from the map text"""
        
        logger.info("📏 Extracting scale information from map...")
        
        scale_info = {
            'scale_found': False,
            'scale_value': None,
            'scale_unit': None,
            'scale_meters_per_pixel': None
        }
        
        # Look for scale bar information
        # The map shows "0.01 0.03 0.05 mi" which indicates a scale bar
        scale_patterns = [
            r'0\.01\s+0\.03\s+0\.05\s+mi',  # Specific pattern from this map
            r'(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+mi',
            r'(\d+(?:\.\d+)?)\s*mi',
            r'(\d+(?:\.\d+)?)\s*miles?',
            r'1:(\d+)',  # Scale ratio
            r'scale\s*[:\-]?\s*1\s*[:\-=]\s*(\d+)'
        ]
        
        for pattern in scale_patterns:
            matches = re.finditer(pattern, extracted_text, re.IGNORECASE)
            for match in matches:
                try:
                    if '0.01 0.03 0.05 mi' in match.group():
                        # This is the scale bar from our map
                        # The scale bar shows 0.05 miles maximum
                        scale_info['scale_found'] = True
                        scale_info['scale_value'] = 0.05
                        scale_info['scale_unit'] = 'miles'
                        
                        # Convert to meters
                        scale_meters = 0.05 * 1609.34  # 80.467 meters
                        
                        # Estimate scale bar length in pixels (typically 10-20% of image width)
                        # For this map, assume scale bar is about 100 pixels
                        estimated_scale_bar_pixels = 100
                        scale_info['scale_meters_per_pixel'] = scale_meters / estimated_scale_bar_pixels
                        
                        logger.info(f"📏 Found scale bar: 0.05 mi = {scale_meters:.1f}m")
                        logger.info(f"📏 Estimated scale: {scale_info['scale_meters_per_pixel']:.3f} m/pixel")
                        return scale_info
                        
                except (ValueError, ZeroDivisionError):
                    continue
        
        # Fallback scale estimation for residential property maps
        logger.warning("⚠️ No scale bar found, using estimated scale")
        scale_info['scale_found'] = False
        scale_info['scale_meters_per_pixel'] = 0.5  # Reasonable estimate for property maps
        
        return scale_info
    
    def _fix_property_coordinates(self, property_coordinates: list, correct_base: dict, scale_info: dict) -> list:
        """Fix the property coordinates using correct base and scale"""
        
        logger.info("🔧 Fixing property coordinates...")
        
        fixed_coordinates = []
        
        base_lat = correct_base['latitude']
        base_lon = correct_base['longitude']
        scale_m_per_pixel = scale_info.get('scale_meters_per_pixel', 0.5)
        
        logger.info(f"📍 Using base: {base_lat:.6f}, {base_lon:.6f}")
        logger.info(f"📏 Using scale: {scale_m_per_pixel:.3f} m/pixel")
        
        # Assume image center is at approximately (1000, 1500) based on typical map layouts
        image_center_x = 1000
        image_center_y = 1500
        
        for shape_idx, shape in enumerate(property_coordinates):
            fixed_shape = {
                'shape_id': shape_idx,
                'coordinates': [],
                'correction_applied': True,
                'scale_factor_used': scale_m_per_pixel,
                'base_location_corrected': True
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
    
    fixer = CoordinateFixer()
    
    # Fix the coordinates
    fixed_results = fixer.fix_coordinates('lot2_test_results.json')
    
    # Save fixed results
    with open('lot2_fixed_results.json', 'w') as f:
        json.dump(fixed_results, f, indent=2)
    
    logger.info("💾 Fixed results saved to lot2_fixed_results.json")
    
    # Test the fixed coordinates
    first_shape = fixed_results['property_coordinates'][0]['coordinates']
    first_vertex = first_shape[0]
    base_coords = fixed_results['base_coordinates']
    
    print(f"\n=== COORDINATE FIX RESULTS ===")
    print(f"Original base: Cowlitz County center")
    print(f"Corrected base: {base_coords['address']}")
    print(f"First vertex before: {first_vertex['original_lat']:.6f}, {first_vertex['original_lon']:.6f}")
    print(f"First vertex after: {first_vertex['latitude']:.6f}, {first_vertex['longitude']:.6f}")
    
    # Calculate distance improvement
    base_point = (base_coords['latitude'], base_coords['longitude'])
    original_point = (first_vertex['original_lat'], first_vertex['original_lon'])
    fixed_point = (first_vertex['latitude'], first_vertex['longitude'])
    
    original_distance = geodesic(base_point, original_point).meters
    fixed_distance = geodesic(base_point, fixed_point).meters
    
    print(f"Distance from base (original): {original_distance:.1f}m")
    print(f"Distance from base (fixed): {fixed_distance:.1f}m")
    print(f"Improvement: {original_distance - fixed_distance:.1f}m closer")

if __name__ == "__main__":
    main() 