#!/usr/bin/env python3
"""
Enhanced Coordinate System v5.0 for Property Map Processing
Major improvements for LOT 2 324 Dolan Road accuracy optimization
"""

import json
import math
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import requests
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCoordinateSystemV5:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="PropertyMapAnalyzer-v5.0")
        
        # Enhanced geocoding strategies for LOT 2 Dolan Road
        self.geocoding_strategies = [
            "324 Dolan Road, Cowlitz County, WA",
            "324 Dolan Rd, Cowlitz County, Washington",  
            "Dolan Road, Cowlitz County, WA",
            "324 Dolan Road, Longview, WA",  # Nearby city
            "324 Dolan Road, Castle Rock, WA",  # Nearby city
            "324 Dolan Road, Washington State",
            "Cowlitz County, Washington"
        ]
        
        # Known reference points for Cowlitz County area
        self.reference_locations = {
            'cowlitz_county_center': (46.158, -122.798),
            'longview_wa': (46.138, -122.938),
            'castle_rock_wa': (46.275, -122.907),
            'kelso_wa': (46.147, -122.908)
        }
        
        # Enhanced scale detection patterns
        self.scale_patterns = [
            r'0\.01\s+0\.03\s+0\.05\s*mi',  # Exact pattern from LOT 2 map
            r'(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s*(mi|miles|ft|feet)',
            r'scale\s*:?\s*1\s*:\s*(\d+)',
            r'(\d+\.?\d*)\s*(mi|miles|ft|feet)\s*=\s*(\d+\.?\d*)\s*(in|inch|px|pixel)',
            r'(\d+\.?\d*)\s*(in|inch)\s*=\s*(\d+\.?\d*)\s*(mi|miles|ft|feet)'
        ]

    def enhanced_geocoding(self, extracted_text: str) -> Dict[str, Any]:
        """Enhanced geocoding with multiple strategies and validation"""
        logger.info("🌍 Starting enhanced geocoding with multiple strategies...")
        
        best_location = None
        best_confidence = 0.0
        geocoding_attempts = []
        
        # Extract specific address information from text
        addresses = self._extract_addresses_advanced(extracted_text)
        logger.info(f"📍 Extracted addresses: {addresses}")
        
        # Try extracted addresses first
        for address in addresses:
            result = self._geocode_with_validation(address)
            if result and result['confidence'] > best_confidence:
                best_location = result
                best_confidence = result['confidence']
            geocoding_attempts.append(result)
        
        # Try predefined strategies
        for strategy in self.geocoding_strategies:
            result = self._geocode_with_validation(strategy)
            if result and result['confidence'] > best_confidence:
                best_location = result
                best_confidence = result['confidence']
            geocoding_attempts.append(result)
            
            # Early exit for high confidence
            if best_confidence > 0.9:
                break
                
        # If still no good match, use reference locations
        if not best_location or best_confidence < 0.5:
            logger.warning("⚠️ Using fallback reference location")
            best_location = {
                'latitude': self.reference_locations['cowlitz_county_center'][0],
                'longitude': self.reference_locations['cowlitz_county_center'][1],
                'address': 'Cowlitz County Center (fallback)',
                'confidence': 0.6,
                'method': 'reference_fallback'
            }
        
        logger.info(f"✅ Best location: {best_location['latitude']:.6f}, {best_location['longitude']:.6f} (confidence: {best_location['confidence']:.2f})")
        
        return {
            'selected_location': best_location,
            'all_attempts': geocoding_attempts,
            'total_attempts': len(geocoding_attempts)
        }

    def _extract_addresses_advanced(self, text: str) -> List[str]:
        """Advanced address extraction from OCR text"""
        addresses = []
        
        # Pattern for LOT 2 324 Dolan Road
        patterns = [
            r'324\s+[Dd]olan\s+[Rr](?:oad|d)',
            r'[Dd]olan\s+[Rr](?:oad|d)',
            r'\d+\s+[Dd]olan\s+[Rr](?:oad|d)',
            r'LOT\s+2.*324.*[Dd]olan',
            r'324\s+[Dd]olan\s+[RrFf](?:[IDd]|oad)'  # OCR might read Road as FID
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', match)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                if cleaned and len(cleaned) > 5:
                    addresses.append(f"{cleaned}, Cowlitz County, WA")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_addresses = []
        for addr in addresses:
            if addr.lower() not in seen:
                seen.add(addr.lower())
                unique_addresses.append(addr)
        
        return unique_addresses[:5]  # Limit to top 5

    def _geocode_with_validation(self, query: str) -> Optional[Dict[str, Any]]:
        """Geocode with Washington state validation"""
        try:
            logger.debug(f"🔍 Geocoding: {query}")
            location = self.geolocator.geocode(query, timeout=10)
            
            if not location:
                return None
                
            # Validate coordinates are in Washington state
            lat, lon = location.latitude, location.longitude
            confidence = self._calculate_location_confidence(lat, lon, location.address, query)
            
            if confidence < 0.3:  # Too low confidence
                return None
                
            return {
                'latitude': lat,
                'longitude': lon,
                'address': location.address,
                'confidence': confidence,
                'method': 'geocoding',
                'query': query
            }
            
        except Exception as e:
            logger.debug(f"Geocoding failed for '{query}': {e}")
            return None

    def _calculate_location_confidence(self, lat: float, lon: float, address: str, query: str) -> float:
        """Calculate confidence score for geocoded location"""
        confidence = 0.0
        
        # Check if in Washington state bounds
        wa_bounds = {'min_lat': 45.543, 'max_lat': 49.002, 'min_lon': -124.844, 'max_lon': -116.915}
        if wa_bounds['min_lat'] <= lat <= wa_bounds['max_lat'] and wa_bounds['min_lon'] <= lon <= wa_bounds['max_lon']:
            confidence += 0.4
        else:
            return 0.0  # Outside Washington, invalid
            
        # Check if in Cowlitz County area (rough bounds)
        cowlitz_bounds = {'min_lat': 45.9, 'max_lat': 46.5, 'min_lon': -123.2, 'max_lon': -122.3}
        if cowlitz_bounds['min_lat'] <= lat <= cowlitz_bounds['max_lat'] and cowlitz_bounds['min_lon'] <= lon <= cowlitz_bounds['max_lon']:
            confidence += 0.3
            
        # Check address for relevant keywords
        address_lower = address.lower()
        if 'cowlitz' in address_lower:
            confidence += 0.2
        if 'dolan' in address_lower:
            confidence += 0.2
        if 'washington' in address_lower or 'wa' in address_lower:
            confidence += 0.1
            
        # Check query match
        query_lower = query.lower()
        if 'dolan' in query_lower and 'dolan' in address_lower:
            confidence += 0.1
        if '324' in query_lower and '324' in address_lower:
            confidence += 0.1
            
        return min(confidence, 1.0)

    def enhanced_scale_detection(self, extracted_text: str, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Enhanced scale detection with multiple methods"""
        logger.info("📏 Starting enhanced scale detection...")
        
        scale_info = {
            'method': 'none',
            'scale_found': False,
            'scale_meters_per_pixel': None,
            'confidence': 0.0,
            'details': {}
        }
        
        # Method 1: Direct pattern matching (highest priority)
        pattern_result = self._detect_scale_from_patterns(extracted_text)
        if pattern_result['scale_found']:
            scale_info.update(pattern_result)
            
        # Method 2: Property size estimation
        if not scale_info['scale_found'] or scale_info['confidence'] < 0.7:
            property_result = self._estimate_scale_from_property_size(image_shape)
            if not scale_info['scale_found'] or property_result['confidence'] > scale_info['confidence']:
                scale_info.update(property_result)
                
        # Method 3: Map features analysis
        if not scale_info['scale_found'] or scale_info['confidence'] < 0.8:
            features_result = self._analyze_map_features_for_scale(extracted_text, image_shape)
            if not scale_info['scale_found'] or features_result['confidence'] > scale_info['confidence']:
                scale_info.update(features_result)
        
        logger.info(f"📏 Scale detection result: {scale_info['scale_meters_per_pixel']:.3f} m/pixel (confidence: {scale_info['confidence']:.2f})")
        return scale_info

    def _detect_scale_from_patterns(self, text: str) -> Dict[str, Any]:
        """Detect scale from text patterns"""
        for pattern in self.scale_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Special handling for LOT 2 map pattern: "0.01 0.03 0.05 mi"
                if '0.01' in match.group(0) and '0.03' in match.group(0) and '0.05' in match.group(0):
                    # This represents a 0.05 mile scale bar
                    scale_miles = 0.05
                    scale_meters = scale_miles * 1609.34  # 80.467 meters
                    
                    # Estimate scale bar length in pixels (typically 80-150 pixels for this type)
                    estimated_pixels = 120
                    meters_per_pixel = scale_meters / estimated_pixels
                    
                    return {
                        'method': 'text_pattern_exact',
                        'scale_found': True,
                        'scale_meters_per_pixel': meters_per_pixel,
                        'confidence': 0.9,
                        'details': {
                            'pattern': match.group(0),
                            'scale_miles': scale_miles,
                            'scale_meters': scale_meters,
                            'estimated_bar_pixels': estimated_pixels
                        }
                    }
        
        return {'method': 'text_pattern', 'scale_found': False, 'confidence': 0.0}

    def _estimate_scale_from_property_size(self, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Estimate scale based on typical residential property size"""
        height, width = image_shape
        
        # Typical residential lot: 0.1 to 2 acres (400-8000 sq meters)
        # Assume lot takes up 20-40% of image
        typical_lot_side_meters = 50  # 50x50m = 2500 sq meters (reasonable lot)
        
        # Estimate what portion of image the lot occupies
        estimated_lot_pixels = min(width, height) * 0.3  # 30% of smaller dimension
        
        meters_per_pixel = typical_lot_side_meters / estimated_lot_pixels
        
        return {
            'method': 'property_size_estimation',
            'scale_found': True,
            'scale_meters_per_pixel': meters_per_pixel,
            'confidence': 0.6,
            'details': {
                'assumed_lot_size_meters': typical_lot_side_meters,
                'estimated_lot_pixels': estimated_lot_pixels,
                'image_dimensions': image_shape
            }
        }

    def _analyze_map_features_for_scale(self, text: str, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze map features to estimate scale"""
        # Look for street names and property numbers to estimate density
        street_count = len(re.findall(r'[Dd]olan|[Jj]uanita|[Mm]ountain', text, re.IGNORECASE))
        property_numbers = len(re.findall(r'\d{3}\s*[Dd]olan', text, re.IGNORECASE))
        
        # More street references = higher density = smaller scale
        if street_count > 5 and property_numbers > 2:
            # High density suburban area
            meters_per_pixel = 0.5
            confidence = 0.7
        elif street_count > 2:
            # Medium density
            meters_per_pixel = 0.7
            confidence = 0.6
        else:
            # Lower density or rural
            meters_per_pixel = 1.0
            confidence = 0.5
            
        return {
            'method': 'map_features_analysis',
            'scale_found': True,
            'scale_meters_per_pixel': meters_per_pixel,
            'confidence': confidence,
            'details': {
                'street_references': street_count,
                'property_numbers': property_numbers
            }
        }

    def enhanced_coordinate_conversion(self, shapes: List[Dict], base_location: Dict, 
                                     scale_info: Dict, image_shape: Tuple[int, int]) -> List[Dict]:
        """Enhanced coordinate conversion with improved accuracy"""
        logger.info("🔧 Starting enhanced coordinate conversion...")
        
        if not shapes:
            return []
            
        base_lat = base_location['latitude']
        base_lon = base_location['longitude']
        meters_per_pixel = scale_info.get('scale_meters_per_pixel', 0.67)
        
        # Calculate property centroid for better reference point
        all_x = []
        all_y = []
        for shape in shapes:
            for coord in shape['coordinates']:
                all_x.append(coord['pixel_x'])
                all_y.append(coord['pixel_y'])
        
        if not all_x:
            return []
            
        centroid_x = sum(all_x) / len(all_x)
        centroid_y = sum(all_y) / len(all_y)
        
        logger.info(f"📍 Property centroid: ({centroid_x:.1f}, {centroid_y:.1f})")
        logger.info(f"📏 Using scale: {meters_per_pixel:.3f} m/pixel")
        
        converted_shapes = []
        
        for idx, shape in enumerate(shapes):
            converted_shape = {
                'shape_id': idx,
                'coordinates': [],
                'conversion_method': 'enhanced_v5',
                'base_location': base_location,
                'scale_info': scale_info,
                'centroid_reference': {'x': centroid_x, 'y': centroid_y}
            }
            
            for coord in shape['coordinates']:
                # Calculate offset from property centroid
                dx_pixels = coord['pixel_x'] - centroid_x
                dy_pixels = coord['pixel_y'] - centroid_y
                
                # Convert to meters
                dx_meters = dx_pixels * meters_per_pixel
                dy_meters = -dy_pixels * meters_per_pixel  # Flip Y for map coordinates
                
                # Calculate new geographic coordinates
                new_lat, new_lon = self._precise_geodetic_offset(
                    base_lat, base_lon, dx_meters, dy_meters
                )
                
                converted_coord = {
                    'latitude': new_lat,
                    'longitude': new_lon,
                    'pixel_x': coord['pixel_x'],
                    'pixel_y': coord['pixel_y'],
                    'offset_meters': {'dx': dx_meters, 'dy': dy_meters},
                    'original_lat': coord.get('latitude', 0),
                    'original_lon': coord.get('longitude', 0)
                }
                
                converted_shape['coordinates'].append(converted_coord)
            
            converted_shapes.append(converted_shape)
            logger.info(f"✅ Converted shape {idx + 1} with {len(converted_shape['coordinates'])} vertices")
        
        return converted_shapes

    def _precise_geodetic_offset(self, base_lat: float, base_lon: float, 
                                dx_meters: float, dy_meters: float) -> Tuple[float, float]:
        """Precise geodetic coordinate offset calculation"""
        # WGS84 ellipsoid parameters
        a = 6378137.0  # Semi-major axis
        f = 1/298.257223563  # Flattening
        
        lat_rad = math.radians(base_lat)
        
        # Calculate meridional radius of curvature (M)
        e2 = 2*f - f*f  # First eccentricity squared
        M = a * (1 - e2) / pow(1 - e2 * math.sin(lat_rad)**2, 1.5)
        
        # Calculate prime vertical radius of curvature (N)
        N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
        
        # Calculate coordinate differences
        dlat = dy_meters / M
        dlon = dx_meters / (N * math.cos(lat_rad))
        
        # Convert back to degrees
        new_lat = base_lat + math.degrees(dlat)
        new_lon = base_lon + math.degrees(dlon)
        
        return new_lat, new_lon

    def comprehensive_accuracy_test(self, results: Dict) -> Dict[str, Any]:
        """Comprehensive accuracy testing with detailed metrics"""
        logger.info("🎯 Starting comprehensive accuracy test...")
        
        if not results.get('property_coordinates'):
            return {'error': 'No property coordinates to test'}
            
        base_coords = results['base_coordinates']
        property_coords = results['property_coordinates'][0]['coordinates']  # Test first shape
        
        test_results = {
            'tests_performed': [],
            'passed_tests': 0,
            'total_tests': 0,
            'overall_score': 0.0,
            'grade': 'F',
            'detailed_metrics': {}
        }
        
        # Test 1: Geographic bounds (Washington State)
        test1 = self._test_geographic_bounds(property_coords)
        test_results['tests_performed'].append(test1)
        if test1['passed']:
            test_results['passed_tests'] += 1
        test_results['total_tests'] += 1
        
        # Test 2: County proximity (Cowlitz County)
        test2 = self._test_county_proximity(property_coords)
        test_results['tests_performed'].append(test2)
        if test2['passed']:
            test_results['passed_tests'] += 1
        test_results['total_tests'] += 1
        
        # Test 3: Distance from base location
        test3 = self._test_base_distance(property_coords, base_coords)
        test_results['tests_performed'].append(test3)
        if test3['passed']:
            test_results['passed_tests'] += 1
        test_results['total_tests'] += 1
        
        # Test 4: Property dimensions
        test4 = self._test_property_dimensions(property_coords)
        test_results['tests_performed'].append(test4)
        if test4['passed']:
            test_results['passed_tests'] += 1
        test_results['total_tests'] += 1
        
        # Test 5: Property area
        test5 = self._test_property_area(property_coords)
        test_results['tests_performed'].append(test5)
        if test5['passed']:
            test_results['passed_tests'] += 1
        test_results['total_tests'] += 1
        
        # Test 6: Coordinate precision
        test6 = self._test_coordinate_precision(property_coords)
        test_results['tests_performed'].append(test6)
        if test6['passed']:
            test_results['passed_tests'] += 1
        test_results['total_tests'] += 1
        
        # Calculate overall score
        test_results['overall_score'] = test_results['passed_tests'] / test_results['total_tests']
        
        # Assign grade
        if test_results['overall_score'] >= 0.9:
            test_results['grade'] = 'A'
        elif test_results['overall_score'] >= 0.8:
            test_results['grade'] = 'B'
        elif test_results['overall_score'] >= 0.7:
            test_results['grade'] = 'C'
        elif test_results['overall_score'] >= 0.6:
            test_results['grade'] = 'D'
        else:
            test_results['grade'] = 'F'
        
        logger.info(f"🎯 Accuracy test complete: {test_results['passed_tests']}/{test_results['total_tests']} ({test_results['overall_score']:.1%}) - Grade {test_results['grade']}")
        return test_results

    def _test_geographic_bounds(self, coords: List[Dict]) -> Dict[str, Any]:
        """Test if coordinates are within Washington state bounds"""
        wa_bounds = {'min_lat': 45.543, 'max_lat': 49.002, 'min_lon': -124.844, 'max_lon': -116.915}
        
        in_bounds = 0
        for coord in coords:
            lat, lon = coord['latitude'], coord['longitude']
            if wa_bounds['min_lat'] <= lat <= wa_bounds['max_lat'] and wa_bounds['min_lon'] <= lon <= wa_bounds['max_lon']:
                in_bounds += 1
        
        percentage = in_bounds / len(coords)
        passed = percentage >= 0.9  # 90% of coordinates must be in WA
        
        return {
            'test_name': 'Geographic Bounds (Washington State)',
            'passed': passed,
            'percentage': percentage,
            'details': f"{in_bounds}/{len(coords)} coordinates in Washington bounds"
        }

    def _test_county_proximity(self, coords: List[Dict]) -> Dict[str, Any]:
        """Test proximity to Cowlitz County area"""
        cowlitz_center = (46.158, -122.798)
        
        close_enough = 0
        distances = []
        
        for coord in coords:
            point = (coord['latitude'], coord['longitude'])
            distance = geodesic(cowlitz_center, point).kilometers
            distances.append(distance)
            if distance <= 50:  # Within 50km of county center
                close_enough += 1
        
        percentage = close_enough / len(coords)
        passed = percentage >= 0.8  # 80% must be within 50km
        avg_distance = sum(distances) / len(distances)
        
        return {
            'test_name': 'County Proximity (Cowlitz County)',
            'passed': passed,
            'percentage': percentage,
            'avg_distance_km': avg_distance,
            'details': f"{close_enough}/{len(coords)} coordinates within 50km of Cowlitz County center"
        }

    def _test_base_distance(self, coords: List[Dict], base_coords: Dict) -> Dict[str, Any]:
        """Test distance from base geocoded location"""
        base_point = (base_coords['latitude'], base_coords['longitude'])
        
        reasonable_distance = 0
        distances = []
        
        for coord in coords:
            point = (coord['latitude'], coord['longitude'])
            distance = geodesic(base_point, point).meters
            distances.append(distance)
            if distance <= 2000:  # Within 2km of base location
                reasonable_distance += 1
        
        percentage = reasonable_distance / len(coords)
        passed = percentage >= 0.7  # 70% must be within 2km
        avg_distance = sum(distances) / len(distances)
        
        return {
            'test_name': 'Base Location Distance',
            'passed': passed,
            'percentage': percentage,
            'avg_distance_meters': avg_distance,
            'details': f"{reasonable_distance}/{len(coords)} coordinates within 2km of base location"
        }

    def _test_property_dimensions(self, coords: List[Dict]) -> Dict[str, Any]:
        """Test property dimensions for reasonableness"""
        lats = [c['latitude'] for c in coords]
        lons = [c['longitude'] for c in coords]
        
        lat_span = max(lats) - min(lats)
        lon_span = max(lons) - min(lons)
        
        # Convert to approximate meters
        lat_span_meters = lat_span * 111000
        lon_span_meters = lon_span * 111000 * math.cos(math.radians(sum(lats)/len(lats))) * 0.99330562
        
        # Reasonable residential property: 20-300 meters on each side
        lat_reasonable = 20 <= lat_span_meters <= 300
        lon_reasonable = 20 <= lon_span_meters <= 300
        
        passed = lat_reasonable and lon_reasonable
        
        return {
            'test_name': 'Property Dimensions',
            'passed': passed,
            'lat_span_meters': lat_span_meters,
            'lon_span_meters': lon_span_meters,
            'details': f"Property span: {lat_span_meters:.1f}m x {lon_span_meters:.1f}m"
        }

    def _test_property_area(self, coords: List[Dict]) -> Dict[str, Any]:
        """Test property area for reasonableness"""
        if len(coords) < 3:
            return {'test_name': 'Property Area', 'passed': False, 'details': 'Insufficient coordinates for area calculation'}
        
        # Calculate area using shoelace formula
        area_sq_meters = self._calculate_polygon_area_meters(coords)
        
        # Reasonable residential property: 200-10000 sq meters (0.05-2.5 acres)
        reasonable_area = 200 <= area_sq_meters <= 10000
        
        return {
            'test_name': 'Property Area',
            'passed': reasonable_area,
            'area_sq_meters': area_sq_meters,
            'area_acres': area_sq_meters / 4047,
            'details': f"Property area: {area_sq_meters:.1f} sq meters ({area_sq_meters/4047:.2f} acres)"
        }

    def _test_coordinate_precision(self, coords: List[Dict]) -> Dict[str, Any]:
        """Test coordinate precision and consistency"""
        # Check for reasonable precision (not too many decimal places, not too few)
        precisions = []
        for coord in coords:
            lat_precision = len(str(coord['latitude']).split('.')[-1])
            lon_precision = len(str(coord['longitude']).split('.')[-1])
            precisions.extend([lat_precision, lon_precision])
        
        avg_precision = sum(precisions) / len(precisions)
        reasonable_precision = 4 <= avg_precision <= 8  # 4-8 decimal places is reasonable
        
        # Check for coordinate clustering (vertices should be close to each other)
        distances_between_vertices = []
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                p1 = (coords[i]['latitude'], coords[i]['longitude'])
                p2 = (coords[j]['latitude'], coords[j]['longitude'])
                dist = geodesic(p1, p2).meters
                distances_between_vertices.append(dist)
        
        if distances_between_vertices:
            avg_vertex_distance = sum(distances_between_vertices) / len(distances_between_vertices)
            reasonable_clustering = avg_vertex_distance <= 100  # Average distance between vertices <= 100m
        else:
            reasonable_clustering = False
            avg_vertex_distance = 0
        
        passed = reasonable_precision and reasonable_clustering
        
        return {
            'test_name': 'Coordinate Precision',
            'passed': passed,
            'avg_precision': avg_precision,
            'avg_vertex_distance': avg_vertex_distance,
            'details': f"Avg precision: {avg_precision:.1f} decimals, avg vertex distance: {avg_vertex_distance:.1f}m"
        }

    def _calculate_polygon_area_meters(self, coords: List[Dict]) -> float:
        """Calculate polygon area in square meters using geographic coordinates"""
        if len(coords) < 3:
            return 0.0
        
        # Convert to UTM for accurate area calculation
        # For Washington state, use UTM zone 10N
        total_area = 0.0
        n = len(coords)
        
        # Use geographic approximation for area calculation
        for i in range(n):
            j = (i + 1) % n
            lat1, lon1 = coords[i]['latitude'], coords[i]['longitude']
            lat2, lon2 = coords[j]['latitude'], coords[j]['longitude']
            
            # Convert to approximate meters
            lat_avg = (lat1 + lat2) / 2
            meters_per_deg_lat = 111132.92
            meters_per_deg_lon = 111132.92 * math.cos(math.radians(lat_avg)) * 0.99330562
            
            x1 = lon1 * meters_per_deg_lon
            y1 = lat1 * meters_per_deg_lat
            x2 = lon2 * meters_per_deg_lon
            y2 = lat2 * meters_per_deg_lat
            
            total_area += (x1 * y2 - x2 * y1)
        
        return abs(total_area) / 2.0

def main():
    """Main function to test the enhanced coordinate system"""
    print("🚀 Enhanced Coordinate System v5.0 - Testing Mode")
    
    # Example usage
    enhanced_system = EnhancedCoordinateSystemV5()
    
    # Test with sample extracted text
    sample_text = """(LOT 2) 324 Dolan Road Aerial Map 0.01 0.03 0.05 mi ADDRESS POINTS 
    Cowlitz County GIS Department, Washington State"""
    
    print("\n🌍 Testing Enhanced Geocoding...")
    geocoding_result = enhanced_system.enhanced_geocoding(sample_text)
    print(f"Selected location: {geocoding_result['selected_location']}")
    
    print("\n📏 Testing Enhanced Scale Detection...")
    scale_result = enhanced_system.enhanced_scale_detection(sample_text, (3000, 2000))
    print(f"Scale result: {scale_result}")
    
    print("\n✅ Enhanced Coordinate System v5.0 ready for deployment!")

if __name__ == "__main__":
    main() 