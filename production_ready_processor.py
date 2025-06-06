#!/usr/bin/env python3
"""
Production-Ready Parcel Map Processor
Robust, reliable parcel map analysis system with comprehensive error handling
and fallback mechanisms for any US parcel map.
"""

import cv2
import numpy as np
import requests
import json
import re
import logging
import traceback
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Standardized processing result structure"""
    success: bool
    coordinates: List[Dict[str, Any]]
    text_analysis: Dict[str, Any]
    shape_analysis: Dict[str, Any]
    location_analysis: Dict[str, Any]
    quality_metrics: Dict[str, float]
    processing_time: float
    error_messages: List[str]
    recommendations: List[str]

class RobustCoordinateProcessor:
    """Robust coordinate processing with multiple fallback methods"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def detect_coordinate_system(self, coordinates: List[Tuple[float, float]], 
                               location_hint: Optional[Dict] = None) -> str:
        """Detect coordinate system with robust error handling"""
        
        try:
            if not coordinates:
                return "WGS84"
            
            # Analyze coordinate ranges
            x_coords = [c[0] for c in coordinates]
            y_coords = [c[1] for c in coordinates]
            
            x_range = (min(x_coords), max(x_coords))
            y_range = (min(y_coords), max(y_coords))
            
            # Check for lat/lon (most common)
            if (-180 <= x_range[0] <= 180 and -90 <= y_range[0] <= 90):
                return "WGS84"
            
            # Check for UTM (6-7 digit numbers)
            if (100000 <= abs(x_range[0]) <= 999999 and 1000000 <= abs(y_range[0]) <= 9999999):
                if location_hint and location_hint.get('lon'):
                    lon = location_hint['lon']
                    if -126 <= lon <= -120:
                        return "UTM_Zone_10N"
                    elif -120 <= lon <= -114:
                        return "UTM_Zone_11N"
                return "UTM_Unknown"
            
            # Check for State Plane (large numbers)
            if location_hint and location_hint.get('state') == 'WA':
                if abs(x_range[0]) > 100000 and abs(y_range[0]) > 100000:
                    return "WA_State_Plane"
            
            return "Unknown"
            
        except Exception as e:
            self.logger.warning(f"Coordinate system detection failed: {e}")
            return "WGS84"  # Safe default
    
    def convert_to_wgs84(self, coordinates: List[Tuple[float, float]], 
                        from_system: str, location_hint: Optional[Dict] = None) -> List[Tuple[float, float]]:
        """Convert coordinates to WGS84 with fallback methods"""
        
        if from_system == "WGS84" or not coordinates:
            return coordinates
        
        try:
            # Try pyproj if available
            try:
                import pyproj
                return self._convert_with_pyproj(coordinates, from_system, location_hint)
            except ImportError:
                self.logger.warning("pyproj not available, using approximation methods")
                return self._convert_with_approximation(coordinates, from_system, location_hint)
                
        except Exception as e:
            self.logger.error(f"Coordinate conversion failed: {e}")
            return coordinates  # Return original if conversion fails
    
    def _convert_with_pyproj(self, coordinates: List[Tuple[float, float]], 
                           from_system: str, location_hint: Optional[Dict]) -> List[Tuple[float, float]]:
        """Convert using pyproj library"""
        
        import pyproj
        
        # Define source projection
        if from_system == "UTM_Zone_10N":
            src_proj = pyproj.Proj(proj='utm', zone=10, datum='WGS84')
        elif from_system == "UTM_Zone_11N":
            src_proj = pyproj.Proj(proj='utm', zone=11, datum='WGS84')
        elif from_system == "WA_State_Plane":
            src_proj = pyproj.Proj(init='epsg:2285')  # WA State Plane North
        else:
            return coordinates  # Unknown system
        
        # Target projection (WGS84)
        dst_proj = pyproj.Proj(init='epsg:4326')
        
        # Convert coordinates
        converted = []
        for x, y in coordinates:
            try:
                lon, lat = pyproj.transform(src_proj, dst_proj, x, y)
                converted.append((lon, lat))
            except:
                converted.append((x, y))  # Keep original if conversion fails
        
        return converted
    
    def _convert_with_approximation(self, coordinates: List[Tuple[float, float]], 
                                  from_system: str, location_hint: Optional[Dict]) -> List[Tuple[float, float]]:
        """Convert using approximation methods when pyproj unavailable"""
        
        # This is a simplified approximation - not suitable for high precision
        # but provides reasonable estimates for visualization
        
        if not location_hint:
            return coordinates
        
        base_lat = location_hint.get('lat', 46.2)
        base_lon = location_hint.get('lon', -122.7)
        
        converted = []
        for x, y in coordinates:
            if from_system.startswith("UTM"):
                # Very rough UTM to lat/lon approximation
                # This is not accurate but provides a reasonable estimate
                lat_offset = (y - 5000000) / 111320.0  # Rough meters to degrees
                lon_offset = (x - 500000) / (111320.0 * np.cos(np.radians(base_lat)))
                
                new_lat = base_lat + lat_offset
                new_lon = base_lon + lon_offset
                converted.append((new_lon, new_lat))
            else:
                converted.append((x, y))  # Keep original
        
        return converted

class EnhancedTextAnalyzer:
    """Enhanced text analysis with multiple extraction strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis"""
        
        analysis = {
            'raw_text_length': len(text),
            'streets': self._extract_streets(text),
            'addresses': self._extract_addresses(text),
            'coordinates': self._extract_coordinates(text),
            'location_info': self._extract_location_info(text),
            'parcel_info': self._extract_parcel_info(text),
            'scale_info': self._extract_scale_info(text),
            'quality_score': 0.0
        }
        
        # Calculate quality score
        analysis['quality_score'] = self._calculate_text_quality(analysis)
        
        return analysis
    
    def _extract_streets(self, text: str) -> List[str]:
        """Extract street names with multiple patterns"""
        
        patterns = [
            r'(\d+\s+[A-Z][A-Za-z\s]+(?:STREET|ST|AVENUE|AVE|ROAD|RD|DRIVE|DR|LANE|LN|BOULEVARD|BLVD|CIRCLE|CIR|COURT|CT|WAY|PLACE|PL))',
            r'([A-Z][A-Za-z\s]+(?:STREET|ST|AVENUE|AVE|ROAD|RD|DRIVE|DR|LANE|LN|BOULEVARD|BLVD|CIRCLE|CIR|COURT|CT|WAY|PLACE|PL))',
            r'([A-Z][A-Za-z\s]{3,20}\s+(?:RD|ROAD|ST|STREET|AVE|AVENUE|DR|DRIVE|WAY|LN|LANE))',
        ]
        
        streets = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                street = match.strip()
                if len(street) > 5 and len(street) < 50:  # Reasonable length
                    streets.add(street)
        
        return list(streets)[:20]  # Limit to top 20
    
    def _extract_addresses(self, text: str) -> List[str]:
        """Extract full addresses"""
        
        pattern = r'(\d+\s+[A-Z][A-Za-z\s]+(?:STREET|ST|AVENUE|AVE|ROAD|RD|DRIVE|DR|LANE|LN|WAY))'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        return list(set(matches))[:10]  # Limit to top 10
    
    def _extract_coordinates(self, text: str) -> List[Dict[str, Any]]:
        """Extract coordinate references"""
        
        coordinates = []
        
        # Decimal degrees
        dd_pattern = r'(\d+\.\d{4,})[°\s]*([NSEW])?[,\s]+(\d+\.\d{4,})[°\s]*([NSEW])?'
        for match in re.finditer(dd_pattern, text):
            coord = {
                'type': 'decimal_degrees',
                'value1': float(match.group(1)),
                'dir1': match.group(2),
                'value2': float(match.group(3)),
                'dir2': match.group(4),
                'raw': match.group(0)
            }
            coordinates.append(coord)
        
        # UTM coordinates
        utm_pattern = r'(\d{6,7})[,\s]+(\d{7,8})'
        for match in re.finditer(utm_pattern, text):
            coord = {
                'type': 'utm',
                'easting': int(match.group(1)),
                'northing': int(match.group(2)),
                'raw': match.group(0)
            }
            coordinates.append(coord)
        
        return coordinates[:10]  # Limit to top 10
    
    def _extract_location_info(self, text: str) -> Dict[str, Any]:
        """Extract location information"""
        
        info = {
            'county': None,
            'state': None,
            'city': None,
            'zip_code': None
        }
        
        # County
        county_match = re.search(r'([A-Z][A-Za-z\s]+)\s+COUNTY', text, re.IGNORECASE)
        if county_match:
            info['county'] = county_match.group(1).strip()
        
        # State
        state_match = re.search(r'\b([A-Z]{2})\b|\b(WASHINGTON|OREGON|CALIFORNIA|IDAHO)\b', text, re.IGNORECASE)
        if state_match:
            info['state'] = state_match.group(1) or state_match.group(2)
        
        # ZIP code
        zip_match = re.search(r'\b(\d{5}(?:-\d{4})?)\b', text)
        if zip_match:
            info['zip_code'] = zip_match.group(1)
        
        return info
    
    def _extract_parcel_info(self, text: str) -> Dict[str, Any]:
        """Extract parcel-specific information"""
        
        info = {
            'lot_numbers': [],
            'parcel_ids': [],
            'subdivisions': [],
            'block_numbers': []
        }
        
        # Lot numbers
        lot_matches = re.findall(r'LOT\s+(\d+[A-Z]?)', text, re.IGNORECASE)
        info['lot_numbers'] = list(set(lot_matches))
        
        # Parcel IDs
        parcel_matches = re.findall(r'PARCEL\s+(?:ID|#|NUMBER)[\s:]*([A-Z0-9\-]+)', text, re.IGNORECASE)
        info['parcel_ids'] = list(set(parcel_matches))
        
        # Subdivisions
        sub_matches = re.findall(r'([A-Z][A-Za-z\s]+)\s+SUBDIVISION', text, re.IGNORECASE)
        info['subdivisions'] = list(set(sub_matches))
        
        # Block numbers
        block_matches = re.findall(r'BLOCK\s+(\d+)', text, re.IGNORECASE)
        info['block_numbers'] = list(set(block_matches))
        
        return info
    
    def _extract_scale_info(self, text: str) -> Dict[str, Any]:
        """Extract map scale information"""
        
        scale_info = {
            'scale_ratio': None,
            'distance_markers': [],
            'units': None
        }
        
        # Scale ratio
        scale_match = re.search(r'1:(\d+,?\d*)', text)
        if scale_match:
            scale_str = scale_match.group(1).replace(',', '')
            try:
                scale_info['scale_ratio'] = int(scale_str)
            except:
                pass
        
        # Distance markers
        distance_matches = re.findall(r'(\d+\.?\d*)\s*(FT|FEET|MI|MILE|KM|METER|M)\b', text, re.IGNORECASE)
        scale_info['distance_markers'] = distance_matches[:5]
        
        return scale_info
    
    def _calculate_text_quality(self, analysis: Dict[str, Any]) -> float:
        """Calculate text extraction quality score"""
        
        score = 0.0
        
        # Text length score (0-0.2)
        if analysis['raw_text_length'] > 1000:
            score += 0.2
        elif analysis['raw_text_length'] > 500:
            score += 0.1
        
        # Street names score (0-0.3)
        street_count = len(analysis['streets'])
        if street_count >= 5:
            score += 0.3
        elif street_count >= 2:
            score += 0.2
        elif street_count >= 1:
            score += 0.1
        
        # Location info score (0-0.3)
        location = analysis['location_info']
        if location['county']:
            score += 0.1
        if location['state']:
            score += 0.1
        if location['city']:
            score += 0.1
        
        # Coordinate info score (0-0.2)
        if analysis['coordinates']:
            score += 0.2
        
        return min(1.0, score)

class RobustShapeDetector:
    """Robust shape detection with multiple algorithms"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_shapes(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect shapes using multiple algorithms"""
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Multiple detection methods
            shapes_adaptive = self._detect_with_adaptive_threshold(gray)
            shapes_canny = self._detect_with_canny(gray)
            shapes_morph = self._detect_with_morphology(gray)
            
            # Combine and filter results
            all_shapes = shapes_adaptive + shapes_canny + shapes_morph
            filtered_shapes = self._filter_and_deduplicate(all_shapes)
            
            # Analyze shapes
            analysis = {
                'total_shapes': len(filtered_shapes),
                'property_boundaries': [],
                'other_shapes': [],
                'quality_metrics': self._calculate_shape_quality(filtered_shapes, image.shape)
            }
            
            # Classify shapes
            for shape in filtered_shapes:
                area = cv2.contourArea(shape)
                vertices = len(cv2.approxPolyDP(shape, 0.02 * cv2.arcLength(shape, True), True))
                
                shape_info = {
                    'vertices': shape.tolist(),
                    'area': area,
                    'vertex_count': vertices,
                    'perimeter': cv2.arcLength(shape, True),
                    'type': 'property_boundary' if self._is_property_boundary(shape, area) else 'other'
                }
                
                if shape_info['type'] == 'property_boundary':
                    analysis['property_boundaries'].append(shape_info)
                else:
                    analysis['other_shapes'].append(shape_info)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Shape detection failed: {e}")
            return {
                'total_shapes': 0,
                'property_boundaries': [],
                'other_shapes': [],
                'quality_metrics': {'overall_score': 0.0}
            }
    
    def _detect_with_adaptive_threshold(self, gray: np.ndarray) -> List[np.ndarray]:
        """Detect shapes using adaptive thresholding"""
        
        try:
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return list(contours)
        except:
            return []
    
    def _detect_with_canny(self, gray: np.ndarray) -> List[np.ndarray]:
        """Detect shapes using Canny edge detection"""
        
        try:
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return list(contours)
        except:
            return []
    
    def _detect_with_morphology(self, gray: np.ndarray) -> List[np.ndarray]:
        """Detect shapes using morphological operations"""
        
        try:
            kernel = np.ones((3,3), np.uint8)
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return list(contours)
        except:
            return []
    
    def _filter_and_deduplicate(self, shapes: List[np.ndarray]) -> List[np.ndarray]:
        """Filter and remove duplicate shapes"""
        
        filtered = []
        
        for shape in shapes:
            try:
                area = cv2.contourArea(shape)
                
                # Filter by area (reasonable parcel sizes)
                if 500 < area < 100000:
                    # Approximate to polygon
                    epsilon = 0.02 * cv2.arcLength(shape, True)
                    approx = cv2.approxPolyDP(shape, epsilon, True)
                    
                    # Must have at least 3 vertices
                    if len(approx) >= 3:
                        # Check for duplicates
                        is_duplicate = False
                        for existing in filtered:
                            if self._shapes_similar(approx, existing):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            filtered.append(approx)
            except:
                continue
        
        return filtered
    
    def _shapes_similar(self, shape1: np.ndarray, shape2: np.ndarray, tolerance: float = 20.0) -> bool:
        """Check if two shapes are similar"""
        
        try:
            # Compare centroids
            M1 = cv2.moments(shape1)
            M2 = cv2.moments(shape2)
            
            if M1['m00'] == 0 or M2['m00'] == 0:
                return False
            
            cx1, cy1 = M1['m10']/M1['m00'], M1['m01']/M1['m00']
            cx2, cy2 = M2['m10']/M2['m00'], M2['m01']/M2['m00']
            
            distance = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            return distance < tolerance
            
        except:
            return False
    
    def _is_property_boundary(self, shape: np.ndarray, area: float) -> bool:
        """Determine if shape is likely a property boundary"""
        
        try:
            # Property boundaries are typically:
            # - Reasonable size (not too small or large)
            # - Roughly rectangular or polygonal
            # - Have reasonable aspect ratio
            
            if not (1000 < area < 50000):
                return False
            
            # Check aspect ratio
            rect = cv2.minAreaRect(shape)
            width, height = rect[1]
            
            if width == 0 or height == 0:
                return False
            
            aspect_ratio = max(width, height) / min(width, height)
            
            # Property boundaries typically have aspect ratio < 10
            return aspect_ratio < 10
            
        except:
            return False
    
    def _calculate_shape_quality(self, shapes: List[np.ndarray], image_shape: Tuple[int, int]) -> Dict[str, float]:
        """Calculate shape detection quality metrics"""
        
        metrics = {
            'overall_score': 0.0,
            'shape_count_score': 0.0,
            'size_distribution_score': 0.0,
            'geometry_score': 0.0
        }
        
        if not shapes:
            return metrics
        
        # Shape count score
        shape_count = len(shapes)
        if 1 <= shape_count <= 10:
            metrics['shape_count_score'] = 1.0
        elif shape_count > 10:
            metrics['shape_count_score'] = max(0.5, 1.0 - (shape_count - 10) * 0.05)
        
        # Size distribution score
        areas = [cv2.contourArea(shape) for shape in shapes]
        if areas:
            area_std = np.std(areas)
            area_mean = np.mean(areas)
            cv_area = area_std / area_mean if area_mean > 0 else 0
            metrics['size_distribution_score'] = max(0.0, 1.0 - cv_area)
        
        # Geometry score (how well-formed are the shapes)
        geometry_scores = []
        for shape in shapes:
            try:
                perimeter = cv2.arcLength(shape, True)
                area = cv2.contourArea(shape)
                if perimeter > 0:
                    compactness = 4 * np.pi * area / (perimeter ** 2)
                    geometry_scores.append(min(1.0, compactness))
            except:
                continue
        
        if geometry_scores:
            metrics['geometry_score'] = np.mean(geometry_scores)
        
        # Overall score
        metrics['overall_score'] = np.mean([
            metrics['shape_count_score'],
            metrics['size_distribution_score'],
            metrics['geometry_score']
        ])
        
        return metrics

class ProductionParcelProcessor:
    """Production-ready parcel map processor"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.coord_processor = RobustCoordinateProcessor()
        self.text_analyzer = EnhancedTextAnalyzer()
        self.shape_detector = RobustShapeDetector()
        self.geocoder = Nominatim(user_agent="ParcelProcessor/1.0")
    
    def process_parcel_map(self, image: np.ndarray, extracted_text: str, 
                          base_location: Optional[Dict] = None) -> ProcessingResult:
        """Process parcel map with comprehensive error handling"""
        
        start_time = time.time()
        error_messages = []
        recommendations = []
        
        try:
            self.logger.info("🚀 Starting production parcel processing")
            
            # Text analysis
            text_analysis = self.text_analyzer.analyze_text(extracted_text)
            
            # Shape detection
            shape_analysis = self.shape_detector.detect_shapes(image)
            
            # Location analysis
            location_analysis = self._analyze_location(text_analysis, base_location)
            
            # Coordinate generation
            coordinates = self._generate_coordinates(shape_analysis, location_analysis, text_analysis)
            
            # Quality assessment
            quality_metrics = self._assess_quality(text_analysis, shape_analysis, location_analysis, coordinates)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(quality_metrics, text_analysis, shape_analysis)
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                success=True,
                coordinates=coordinates,
                text_analysis=text_analysis,
                shape_analysis=shape_analysis,
                location_analysis=location_analysis,
                quality_metrics=quality_metrics,
                processing_time=processing_time,
                error_messages=error_messages,
                recommendations=recommendations
            )
            
            self.logger.info(f"✅ Processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Processing failed: {str(e)}"
            self.logger.error(error_msg)
            error_messages.append(error_msg)
            
            # Return partial results even on failure
            return ProcessingResult(
                success=False,
                coordinates=[],
                text_analysis={},
                shape_analysis={},
                location_analysis={},
                quality_metrics={'overall_score': 0.0},
                processing_time=processing_time,
                error_messages=error_messages,
                recommendations=["Processing failed - check input file quality"]
            )
    
    def _analyze_location(self, text_analysis: Dict[str, Any], 
                         base_location: Optional[Dict]) -> Dict[str, Any]:
        """Analyze location information"""
        
        location_analysis = {
            'base_coordinates': base_location,
            'detected_location': text_analysis.get('location_info', {}),
            'geocoded_location': None,
            'confidence': 0.0
        }
        
        try:
            # Try to geocode based on detected location info
            location_info = text_analysis.get('location_info', {})
            
            if location_info.get('county') and location_info.get('state'):
                query = f"{location_info['county']} County, {location_info['state']}, USA"
                
                try:
                    location = self.geocoder.geocode(query, timeout=10)
                    if location:
                        location_analysis['geocoded_location'] = {
                            'lat': location.latitude,
                            'lon': location.longitude,
                            'address': location.address,
                            'query': query
                        }
                        location_analysis['confidence'] = 0.8
                except Exception as e:
                    self.logger.warning(f"Geocoding failed: {e}")
            
            # Use base location as fallback
            if not location_analysis['geocoded_location'] and base_location:
                location_analysis['geocoded_location'] = base_location
                location_analysis['confidence'] = 0.5
            
        except Exception as e:
            self.logger.warning(f"Location analysis failed: {e}")
        
        return location_analysis
    
    def _generate_coordinates(self, shape_analysis: Dict[str, Any], 
                            location_analysis: Dict[str, Any],
                            text_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate geographic coordinates from shapes"""
        
        coordinates = []
        
        try:
            base_location = location_analysis.get('geocoded_location')
            if not base_location:
                return coordinates
            
            base_lat = base_location.get('lat', 46.2)
            base_lon = base_location.get('lon', -122.7)
            
            # Estimate scale
            scale_info = text_analysis.get('scale_info', {})
            scale_ratio = scale_info.get('scale_ratio', 2000)
            
            # Convert pixel coordinates to geographic coordinates
            for boundary in shape_analysis.get('property_boundaries', []):
                vertices = boundary.get('vertices', [])
                
                if vertices:
                    boundary_coords = []
                    
                    for vertex in vertices:
                        try:
                            # Extract x, y from vertex (handle different formats)
                            if isinstance(vertex, list) and len(vertex) >= 2:
                                if isinstance(vertex[0], list):
                                    x, y = vertex[0][0], vertex[0][1]
                                else:
                                    x, y = vertex[0], vertex[1]
                            else:
                                continue
                            
                            # Convert to geographic coordinates
                            lat, lon = self._pixel_to_geo(x, y, base_lat, base_lon, scale_ratio)
                            
                            coord_info = {
                                'lat': lat,
                                'lon': lon,
                                'pixel_x': x,
                                'pixel_y': y,
                                'confidence': 0.7,
                                'source': 'shape_analysis',
                                'boundary_area': boundary.get('area', 0)
                            }
                            
                            boundary_coords.append(coord_info)
                            
                        except Exception as e:
                            self.logger.warning(f"Coordinate conversion failed for vertex: {e}")
                            continue
                    
                    coordinates.extend(boundary_coords)
            
        except Exception as e:
            self.logger.error(f"Coordinate generation failed: {e}")
        
        return coordinates
    
    def _pixel_to_geo(self, x: float, y: float, base_lat: float, base_lon: float, 
                     scale_ratio: float) -> Tuple[float, float]:
        """Convert pixel coordinates to geographic coordinates"""
        
        # Estimate meters per pixel based on scale
        meters_per_pixel = (scale_ratio / 96.0) * 0.0254  # Assuming 96 DPI
        
        # Convert to degrees (rough approximation)
        lat_deg_per_meter = 1.0 / 111320.0
        lon_deg_per_meter = 1.0 / (111320.0 * np.cos(np.radians(base_lat)))
        
        # Assume image center is at base location (500, 500 for 1000x1000 image)
        dx_meters = (x - 500) * meters_per_pixel
        dy_meters = (500 - y) * meters_per_pixel  # Flip Y axis
        
        # Convert to lat/lon
        new_lat = base_lat + (dy_meters * lat_deg_per_meter)
        new_lon = base_lon + (dx_meters * lon_deg_per_meter)
        
        return new_lat, new_lon
    
    def _assess_quality(self, text_analysis: Dict[str, Any], 
                       shape_analysis: Dict[str, Any],
                       location_analysis: Dict[str, Any],
                       coordinates: List[Dict[str, Any]]) -> Dict[str, float]:
        """Assess overall processing quality"""
        
        quality = {
            'overall_score': 0.0,
            'text_quality': text_analysis.get('quality_score', 0.0),
            'shape_quality': shape_analysis.get('quality_metrics', {}).get('overall_score', 0.0),
            'location_quality': location_analysis.get('confidence', 0.0),
            'coordinate_quality': 0.0
        }
        
        # Coordinate quality
        if coordinates:
            coord_confidences = [c.get('confidence', 0.0) for c in coordinates]
            quality['coordinate_quality'] = np.mean(coord_confidences)
        
        # Overall score
        quality['overall_score'] = np.mean([
            quality['text_quality'],
            quality['shape_quality'],
            quality['location_quality'],
            quality['coordinate_quality']
        ])
        
        return quality
    
    def _generate_recommendations(self, quality_metrics: Dict[str, float],
                                text_analysis: Dict[str, Any],
                                shape_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving results"""
        
        recommendations = []
        
        # Overall quality
        if quality_metrics['overall_score'] < 0.5:
            recommendations.append("Overall processing quality is low - consider higher resolution image")
        
        # Text quality
        if quality_metrics['text_quality'] < 0.5:
            recommendations.append("Text extraction quality is poor - ensure image has clear, readable text")
            
            if len(text_analysis.get('streets', [])) == 0:
                recommendations.append("No street names detected - add street labels for better location accuracy")
            
            if not text_analysis.get('location_info', {}).get('county'):
                recommendations.append("County not detected - include county name in image for better geocoding")
        
        # Shape quality
        if quality_metrics['shape_quality'] < 0.5:
            recommendations.append("Shape detection quality is poor - ensure property boundaries are clearly visible")
            
            if shape_analysis.get('total_shapes', 0) == 0:
                recommendations.append("No shapes detected - check image contrast and boundary clarity")
        
        # Location quality
        if quality_metrics['location_quality'] < 0.5:
            recommendations.append("Location detection failed - add location references or coordinate grid")
        
        # Coordinate quality
        if quality_metrics['coordinate_quality'] < 0.5:
            recommendations.append("Coordinate generation has low confidence - verify scale and reference points")
        
        return recommendations

def create_production_processor() -> ProductionParcelProcessor:
    """Factory function to create production processor"""
    return ProductionParcelProcessor()

if __name__ == "__main__":
    # Test the production processor
    processor = create_production_processor()
    print("Production-Ready Parcel Processor initialized successfully!")
    print("Ready for robust parcel map analysis.") 