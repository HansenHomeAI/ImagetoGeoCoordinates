import logging
import numpy as np
import json
import math
from typing import List, Dict, Any, Tuple, Optional
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import requests
import time
import os

logger = logging.getLogger(__name__)

class CoordinateAccuracyTester:
    """Comprehensive testing system for coordinate accuracy validation"""
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="accuracy_tester", timeout=10)
        
        # Known reference points for validation
        self.reference_points = {
            '324_dolan_road': {
                'expected_area': {'lat': 46.096, 'lon': -122.621},  # Cowlitz County center
                'address_patterns': ['324 dolan road', 'dolan road', 'dolan'],
                'county': 'cowlitz',
                'state': 'washington'
            }
        }
        
        # Validation thresholds
        self.thresholds = {
            'max_distance_from_address': 5000,  # meters
            'min_property_area': 100,  # square meters
            'max_property_area': 10000,  # square meters
            'reasonable_coordinate_precision': 6,  # decimal places
            'max_coordinate_cluster_span': 1000  # meters
        }
    
    def run_comprehensive_test(self, coordinate_data: Dict[str, Any], 
                             extracted_text: str, test_image_path: str) -> Dict[str, Any]:
        """Run comprehensive coordinate accuracy testing"""
        
        logger.info("🧪 Starting comprehensive coordinate accuracy test...")
        
        test_results = {
            'test_timestamp': time.time(),
            'test_image': test_image_path,
            'input_data_summary': self._summarize_input_data(coordinate_data),
            'tests': {}
        }
        
        # Test 1: Geographic Reasonableness
        test_results['tests']['geographic_reasonableness'] = self._test_geographic_reasonableness(
            coordinate_data, extracted_text
        )
        
        # Test 2: Distance Validation
        test_results['tests']['distance_validation'] = self._test_distance_validation(
            coordinate_data, extracted_text
        )
        
        # Test 3: Property Size Validation
        test_results['tests']['property_size_validation'] = self._test_property_size_validation(
            coordinate_data
        )
        
        # Test 4: Coordinate Clustering Analysis
        test_results['tests']['clustering_analysis'] = self._test_coordinate_clustering(
            coordinate_data
        )
        
        # Test 5: Reverse Geocoding Validation
        test_results['tests']['reverse_geocoding'] = self._test_reverse_geocoding(
            coordinate_data, extracted_text
        )
        
        # Test 6: Scale Consistency Check
        test_results['tests']['scale_consistency'] = self._test_scale_consistency(
            coordinate_data
        )
        
        # Calculate overall accuracy score
        test_results['overall_accuracy'] = self._calculate_overall_accuracy(test_results['tests'])
        
        # Generate recommendations
        test_results['recommendations'] = self._generate_recommendations(test_results)
        
        return test_results
    
    def _summarize_input_data(self, coordinate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize input coordinate data"""
        
        summary = {
            'total_coordinate_sets': 0,
            'total_points': 0,
            'coordinate_bounds': None,
            'has_geocoding_info': False,
            'has_shape_info': False
        }
        
        if 'coordinate_conversion' in coordinate_data:
            coord_sets = coordinate_data['coordinate_conversion'].get('coordinate_sets', [])
            summary['total_coordinate_sets'] = len(coord_sets)
            
            all_lats, all_lons = [], []
            for coord_set in coord_sets:
                # Handle both list format and dict format
                if isinstance(coord_set, list):
                    coords = coord_set
                else:
                    coords = coord_set.get('coordinates', [])
                
                summary['total_points'] += len(coords)
                
                for coord in coords:
                    all_lats.append(coord['latitude'])
                    all_lons.append(coord['longitude'])
            
            if all_lats and all_lons:
                summary['coordinate_bounds'] = {
                    'lat_min': min(all_lats), 'lat_max': max(all_lats),
                    'lon_min': min(all_lons), 'lon_max': max(all_lons),
                    'lat_span': max(all_lats) - min(all_lats),
                    'lon_span': max(all_lons) - min(all_lons)
                }
        
        summary['has_geocoding_info'] = 'geocoding' in coordinate_data
        summary['has_shape_info'] = 'shape_detection' in coordinate_data
        
        return summary
    
    def _test_geographic_reasonableness(self, coordinate_data: Dict, extracted_text: str) -> Dict[str, Any]:
        """Test if coordinates are geographically reasonable"""
        
        test_result = {
            'test_name': 'Geographic Reasonableness',
            'passed': False,
            'score': 0.0,
            'details': {},
            'issues': []
        }
        
        try:
            if 'coordinate_conversion' not in coordinate_data:
                test_result['issues'].append("No coordinate data found")
                return test_result
            
            coord_sets = coordinate_data['coordinate_conversion'].get('coordinate_sets', [])
            if not coord_sets:
                test_result['issues'].append("No coordinate sets found")
                return test_result
            
            # Check if coordinates are in Washington State
            wa_bounds = {'min_lat': 45.543, 'max_lat': 49.002, 'min_lon': -124.844, 'max_lon': -116.915}
            
            in_washington_count = 0
            total_points = 0
            
            for coord_set in coord_sets:
                # Handle both list format and dict format
                if isinstance(coord_set, list):
                    coords = coord_set
                else:
                    coords = coord_set.get('coordinates', [])
                
                for coord in coords:
                    total_points += 1
                    lat, lon = coord['latitude'], coord['longitude']
                    
                    if (wa_bounds['min_lat'] <= lat <= wa_bounds['max_lat'] and
                        wa_bounds['min_lon'] <= lon <= wa_bounds['max_lon']):
                        in_washington_count += 1
            
            if total_points > 0:
                wa_percentage = (in_washington_count / total_points) * 100
                test_result['details']['washington_state_percentage'] = wa_percentage
                test_result['details']['points_in_washington'] = in_washington_count
                test_result['details']['total_points'] = total_points
                
                if wa_percentage >= 95:
                    test_result['score'] = 1.0
                    test_result['passed'] = True
                elif wa_percentage >= 50:
                    test_result['score'] = 0.5
                    test_result['issues'].append(f"Only {wa_percentage:.1f}% of points in Washington State")
                else:
                    test_result['score'] = 0.0
                    test_result['issues'].append(f"Only {wa_percentage:.1f}% of points in Washington State - major location error")
            
        except Exception as e:
            test_result['issues'].append(f"Error during geographic test: {str(e)}")
        
        return test_result
    
    def _test_distance_validation(self, coordinate_data: Dict, extracted_text: str) -> Dict[str, Any]:
        """Test if coordinates are reasonable distance from known address"""
        
        test_result = {
            'test_name': 'Distance Validation',
            'passed': False,
            'score': 0.0,
            'details': {},
            'issues': []
        }
        
        try:
            # Get the base geocoded location
            if 'geocoding' not in coordinate_data:
                test_result['issues'].append("No geocoding information available")
                return test_result
            
            base_location = coordinate_data['geocoding']
            base_lat, base_lon = base_location['latitude'], base_location['longitude']
            
            # Get coordinate centroid
            coord_sets = coordinate_data.get('coordinate_conversion', {}).get('coordinate_sets', [])
            if not coord_sets:
                test_result['issues'].append("No coordinate sets to validate")
                return test_result
            
            all_lats, all_lons = [], []
            for coord_set in coord_sets:
                # Handle both list format and dict format
                if isinstance(coord_set, list):
                    coords = coord_set
                else:
                    coords = coord_set.get('coordinates', [])
                
                for coord in coords:
                    all_lats.append(coord['latitude'])
                    all_lons.append(coord['longitude'])
            
            if not all_lats:
                test_result['issues'].append("No coordinates found")
                return test_result
            
            centroid_lat = sum(all_lats) / len(all_lats)
            centroid_lon = sum(all_lons) / len(all_lons)
            
            # Calculate distance
            distance_meters = geodesic((base_lat, base_lon), (centroid_lat, centroid_lon)).meters
            
            test_result['details']['base_location'] = {'lat': base_lat, 'lon': base_lon}
            test_result['details']['coordinate_centroid'] = {'lat': centroid_lat, 'lon': centroid_lon}
            test_result['details']['distance_meters'] = distance_meters
            
            # Score based on distance
            if distance_meters <= 500:  # Within 500m is excellent
                test_result['score'] = 1.0
                test_result['passed'] = True
            elif distance_meters <= 2000:  # Within 2km is acceptable
                test_result['score'] = 0.7
                test_result['passed'] = True
                test_result['issues'].append(f"Coordinates {distance_meters:.0f}m from geocoded address")
            elif distance_meters <= 5000:  # Within 5km is questionable
                test_result['score'] = 0.3
                test_result['issues'].append(f"Coordinates {distance_meters:.0f}m from geocoded address - too far")
            else:  # Beyond 5km is likely wrong
                test_result['score'] = 0.0
                test_result['issues'].append(f"Coordinates {distance_meters:.0f}m from geocoded address - likely incorrect")
            
        except Exception as e:
            test_result['issues'].append(f"Error during distance validation: {str(e)}")
        
        return test_result
    
    def _test_property_size_validation(self, coordinate_data: Dict) -> Dict[str, Any]:
        """Test if calculated property areas are reasonable"""
        
        test_result = {
            'test_name': 'Property Size Validation',
            'passed': False,
            'score': 0.0,
            'details': {},
            'issues': []
        }
        
        try:
            coord_sets = coordinate_data.get('coordinate_conversion', {}).get('coordinate_sets', [])
            if not coord_sets:
                test_result['issues'].append("No coordinate sets found")
                return test_result
            
            property_areas = []
            
            for i, coord_set in enumerate(coord_sets):
                # Handle both list format and dict format
                if isinstance(coord_set, list):
                    coords = coord_set
                else:
                    coords = coord_set.get('coordinates', [])
                
                if len(coords) < 3:
                    continue
                
                # Calculate area using Shoelace formula
                area_m2 = self._calculate_polygon_area(coords)
                area_acres = area_m2 * 0.000247105
                
                property_areas.append({
                    'shape_id': i,
                    'area_m2': area_m2,
                    'area_acres': area_acres,
                    'reasonable': self.thresholds['min_property_area'] <= area_m2 <= self.thresholds['max_property_area']
                })
            
            test_result['details']['property_areas'] = property_areas
            
            if property_areas:
                reasonable_count = sum(1 for area in property_areas if area['reasonable'])
                reasonableness_ratio = reasonable_count / len(property_areas)
                
                test_result['details']['reasonable_properties'] = reasonable_count
                test_result['details']['total_properties'] = len(property_areas)
                test_result['details']['reasonableness_ratio'] = reasonableness_ratio
                
                test_result['score'] = reasonableness_ratio
                test_result['passed'] = reasonableness_ratio >= 0.5
                
                if reasonableness_ratio < 0.5:
                    test_result['issues'].append(f"Only {reasonableness_ratio:.1%} of properties have reasonable areas")
                
                # Check for extremely small areas (scale issues)
                tiny_areas = [area for area in property_areas if area['area_m2'] < 10]
                if tiny_areas:
                    test_result['issues'].append(f"{len(tiny_areas)} properties have areas < 10m² (likely scale issue)")
            else:
                test_result['issues'].append("No valid property areas calculated")
        
        except Exception as e:
            test_result['issues'].append(f"Error during property size validation: {str(e)}")
        
        return test_result
    
    def _test_coordinate_clustering(self, coordinate_data: Dict) -> Dict[str, Any]:
        """Test if coordinates are appropriately clustered"""
        
        test_result = {
            'test_name': 'Coordinate Clustering Analysis',
            'passed': False,
            'score': 0.0,
            'details': {},
            'issues': []
        }
        
        try:
            coord_sets = coordinate_data.get('coordinate_conversion', {}).get('coordinate_sets', [])
            if not coord_sets:
                test_result['issues'].append("No coordinate sets found")
                return test_result
            
            all_lats, all_lons = [], []
            for coord_set in coord_sets:
                # Handle both list format and dict format
                if isinstance(coord_set, list):
                    coords = coord_set
                else:
                    coords = coord_set.get('coordinates', [])
                
                for coord in coords:
                    all_lats.append(coord['latitude'])
                    all_lons.append(coord['longitude'])
            
            if len(all_lats) < 2:
                test_result['issues'].append("Insufficient coordinates for clustering analysis")
                return test_result
            
            # Calculate coordinate spread
            lat_span = max(all_lats) - min(all_lats)
            lon_span = max(all_lons) - min(all_lons)
            
            # Convert to approximate meters
            avg_lat = sum(all_lats) / len(all_lats)
            lat_span_meters = lat_span * 111000
            lon_span_meters = lon_span * 111000 * math.cos(math.radians(avg_lat))
            max_span_meters = max(lat_span_meters, lon_span_meters)
            
            test_result['details']['coordinate_span'] = {
                'lat_span_degrees': lat_span,
                'lon_span_degrees': lon_span,
                'lat_span_meters': lat_span_meters,
                'lon_span_meters': lon_span_meters,
                'max_span_meters': max_span_meters
            }
            
            # Score based on clustering appropriateness
            if 10 <= max_span_meters <= 500:  # Good clustering for property map
                test_result['score'] = 1.0
                test_result['passed'] = True
            elif 500 < max_span_meters <= 1000:  # Acceptable
                test_result['score'] = 0.7
                test_result['passed'] = True
            elif max_span_meters < 10:  # Too tightly clustered (scale issue)
                test_result['score'] = 0.2
                test_result['issues'].append(f"Coordinates too tightly clustered ({max_span_meters:.1f}m span) - likely scale issue")
            else:  # Too spread out
                test_result['score'] = 0.1
                test_result['issues'].append(f"Coordinates too spread out ({max_span_meters:.0f}m span) - likely location issue")
        
        except Exception as e:
            test_result['issues'].append(f"Error during clustering analysis: {str(e)}")
        
        return test_result
    
    def _test_reverse_geocoding(self, coordinate_data: Dict, extracted_text: str) -> Dict[str, Any]:
        """Test coordinates using reverse geocoding"""
        
        test_result = {
            'test_name': 'Reverse Geocoding Validation',
            'passed': False,
            'score': 0.0,
            'details': {},
            'issues': []
        }
        
        try:
            coord_sets = coordinate_data.get('coordinate_conversion', {}).get('coordinate_sets', [])
            if not coord_sets:
                test_result['issues'].append("No coordinate sets found")
                return test_result
            
            # Get centroid of all coordinates
            all_lats, all_lons = [], []
            for coord_set in coord_sets:
                # Handle both list format and dict format
                if isinstance(coord_set, list):
                    coords = coord_set
                else:
                    coords = coord_set.get('coordinates', [])
                
                for coord in coords:
                    all_lats.append(coord['latitude'])
                    all_lons.append(coord['longitude'])
            
            if not all_lats:
                test_result['issues'].append("No coordinates found")
                return test_result
            
            centroid_lat = sum(all_lats) / len(all_lats)
            centroid_lon = sum(all_lons) / len(all_lons)
            
            # Reverse geocode the centroid
            try:
                location = self.geolocator.reverse((centroid_lat, centroid_lon), timeout=10)
                
                if location:
                    reverse_address = location.address.lower()
                    test_result['details']['reverse_geocoded_address'] = location.address
                    test_result['details']['centroid_coordinates'] = {'lat': centroid_lat, 'lon': centroid_lon}
                    
                    # Check for matches with expected location elements
                    score = 0.0
                    matches = []
                    
                    if 'washington' in reverse_address or 'wa' in reverse_address:
                        score += 0.3
                        matches.append('Washington State')
                    
                    if 'cowlitz' in reverse_address:
                        score += 0.4
                        matches.append('Cowlitz County')
                    elif 'skamania' in reverse_address:
                        score += 0.2
                        matches.append('Skamania County')
                    
                    if 'dolan' in reverse_address:
                        score += 0.3
                        matches.append('Dolan Road')
                    
                    test_result['details']['location_matches'] = matches
                    test_result['score'] = min(1.0, score)
                    test_result['passed'] = score >= 0.5
                    
                    if score < 0.5:
                        test_result['issues'].append(f"Reverse geocoding doesn't match expected location elements")
                
                else:
                    test_result['issues'].append("Reverse geocoding failed - no address found")
                
            except Exception as e:
                test_result['issues'].append(f"Reverse geocoding error: {str(e)}")
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            test_result['issues'].append(f"Error during reverse geocoding test: {str(e)}")
        
        return test_result
    
    def _test_scale_consistency(self, coordinate_data: Dict) -> Dict[str, Any]:
        """Test consistency of scale across different measurements"""
        
        test_result = {
            'test_name': 'Scale Consistency Check',
            'passed': False,
            'score': 0.0,
            'details': {},
            'issues': []
        }
        
        try:
            coord_sets = coordinate_data.get('coordinate_conversion', {}).get('coordinate_sets', [])
            if len(coord_sets) < 2:
                test_result['issues'].append("Need at least 2 shapes for scale consistency check")
                return test_result
            
            # Calculate areas and perimeters for each shape
            shape_metrics = []
            
            for i, coord_set in enumerate(coord_sets):
                # Handle both list format and dict format
                if isinstance(coord_set, list):
                    coords = coord_set
                else:
                    coords = coord_set.get('coordinates', [])
                
                if len(coords) < 3:
                    continue
                
                area_m2 = self._calculate_polygon_area(coords)
                perimeter_m = self._calculate_polygon_perimeter(coords)
                
                shape_metrics.append({
                    'shape_id': i,
                    'area_m2': area_m2,
                    'perimeter_m': perimeter_m,
                    'area_to_perimeter_ratio': area_m2 / perimeter_m if perimeter_m > 0 else 0
                })
            
            if len(shape_metrics) < 2:
                test_result['issues'].append("Insufficient valid shapes for consistency analysis")
                return test_result
            
            test_result['details']['shape_metrics'] = shape_metrics
            
            # Check consistency of area-to-perimeter ratios
            ratios = [m['area_to_perimeter_ratio'] for m in shape_metrics if m['area_to_perimeter_ratio'] > 0]
            
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                ratio_variance = sum((r - avg_ratio) ** 2 for r in ratios) / len(ratios)
                ratio_std = math.sqrt(ratio_variance)
                coefficient_of_variation = ratio_std / avg_ratio if avg_ratio > 0 else float('inf')
                
                test_result['details']['ratio_statistics'] = {
                    'average_ratio': avg_ratio,
                    'standard_deviation': ratio_std,
                    'coefficient_of_variation': coefficient_of_variation
                }
                
                # Score based on consistency (lower CV is better)
                if coefficient_of_variation <= 0.5:
                    test_result['score'] = 1.0
                    test_result['passed'] = True
                elif coefficient_of_variation <= 1.0:
                    test_result['score'] = 0.7
                    test_result['passed'] = True
                else:
                    test_result['score'] = 0.3
                    test_result['issues'].append(f"High variability in shape metrics (CV: {coefficient_of_variation:.2f})")
        
        except Exception as e:
            test_result['issues'].append(f"Error during scale consistency check: {str(e)}")
        
        return test_result
    
    def _calculate_polygon_area(self, coordinates: List[Dict]) -> float:
        """Calculate polygon area using Shoelace formula"""
        
        if len(coordinates) < 3:
            return 0.0
        
        try:
            total_area = 0.0
            n = len(coordinates)
            
            for i in range(n):
                j = (i + 1) % n
                
                lat1, lon1 = coordinates[i]['latitude'], coordinates[i]['longitude']
                lat2, lon2 = coordinates[j]['latitude'], coordinates[j]['longitude']
                
                # Convert to approximate Cartesian
                x1 = lon1 * 111320 * math.cos(math.radians(lat1))
                y1 = lat1 * 110540
                x2 = lon2 * 111320 * math.cos(math.radians(lat2))
                y2 = lat2 * 110540
                
                total_area += x1 * y2 - x2 * y1
            
            return abs(total_area) / 2.0
            
        except Exception:
            return 0.0
    
    def _calculate_polygon_perimeter(self, coordinates: List[Dict]) -> float:
        """Calculate polygon perimeter"""
        
        if len(coordinates) < 2:
            return 0.0
        
        try:
            perimeter = 0.0
            n = len(coordinates)
            
            for i in range(n):
                j = (i + 1) % n
                
                point1 = (coordinates[i]['latitude'], coordinates[i]['longitude'])
                point2 = (coordinates[j]['latitude'], coordinates[j]['longitude'])
                
                distance = geodesic(point1, point2).meters
                perimeter += distance
            
            return perimeter
            
        except Exception:
            return 0.0
    
    def _calculate_overall_accuracy(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall accuracy score"""
        
        test_scores = []
        test_weights = {
            'geographic_reasonableness': 0.25,
            'distance_validation': 0.20,
            'property_size_validation': 0.20,
            'clustering_analysis': 0.15,
            'reverse_geocoding': 0.15,
            'scale_consistency': 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for test_name, weight in test_weights.items():
            if test_name in test_results:
                score = test_results[test_name].get('score', 0.0)
                weighted_score += score * weight
                total_weight += weight
                test_scores.append({'test': test_name, 'score': score, 'weight': weight})
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        return {
            'overall_score': overall_score,
            'grade': self._score_to_grade(overall_score),
            'test_scores': test_scores,
            'weighted_calculation': {'numerator': weighted_score, 'denominator': total_weight}
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        # Check individual test results
        tests = test_results.get('tests', {})
        
        if 'geographic_reasonableness' in tests and not tests['geographic_reasonableness'].get('passed', False):
            recommendations.append("❌ CRITICAL: Coordinates are outside expected geographic region - check base location geocoding")
        
        if 'distance_validation' in tests and tests['distance_validation'].get('score', 0) < 0.5:
            recommendations.append("⚠️ High priority: Coordinates are too far from geocoded address - verify scale calculation")
        
        if 'property_size_validation' in tests and tests['property_size_validation'].get('score', 0) < 0.5:
            recommendations.append("⚠️ Property areas are unreasonable - check scale and coordinate transformation math")
        
        if 'clustering_analysis' in tests:
            clustering_details = tests['clustering_analysis'].get('details', {})
            span = clustering_details.get('coordinate_span', {}).get('max_span_meters', 0)
            if span < 10:
                recommendations.append("🔧 Scale issue: Coordinates too tightly clustered - increase scale factor")
            elif span > 1000:
                recommendations.append("🔧 Location issue: Coordinates too spread out - check base location accuracy")
        
        overall_score = test_results.get('overall_accuracy', {}).get('overall_score', 0)
        if overall_score < 0.7:
            recommendations.append("🚨 Overall accuracy is low - comprehensive coordinate system review needed")
        
        return recommendations

def create_accuracy_tester() -> CoordinateAccuracyTester:
    """Factory function to create accuracy tester"""
    return CoordinateAccuracyTester()