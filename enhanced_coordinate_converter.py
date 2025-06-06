#!/usr/bin/env python3
"""
Enhanced Coordinate Conversion for Parcel Maps
Converts pixel coordinates to geo-coordinates using advanced scaling and reference point strategies
"""

import numpy as np
import cv2
import logging
import re
from typing import List, Dict, Tuple, Optional, Any
import math
from geopy.distance import geodesic

logger = logging.getLogger(__name__)

class EnhancedCoordinateConverter:
    """Enhanced coordinate conversion with multiple scaling strategies"""
    
    def __init__(self):
        self.scale_cache = {}
        
    def convert_shapes_to_coordinates(self, 
                                    shapes: List[np.ndarray], 
                                    base_location: Dict[str, Any],
                                    image_shape: Tuple[int, int],
                                    extracted_text: str) -> List[Dict[str, Any]]:
        """Convert detected shapes to geo-coordinates using multiple strategies"""
        
        logger.info(f"🌍 Converting {len(shapes)} shapes to geo-coordinates...")
        
        if not shapes:
            logger.warning("⚠️ No shapes provided for coordinate conversion")
            return []
        
        if not base_location:
            logger.warning("⚠️ No base location provided for coordinate conversion")
            return []
        
        # Extract scale information from text and image
        scale_info = self._extract_scale_information(extracted_text, image_shape)
        logger.info(f"📏 Scale information: {scale_info}")
        
        # Calculate base coordinates
        base_lat = base_location.get('latitude')
        base_lon = base_location.get('longitude')
        
        if not base_lat or not base_lon:
            logger.error("❌ Invalid base coordinates")
            return []
        
        # Convert each shape using the best available method
        coordinate_sets = []
        
        for i, shape in enumerate(shapes):
            logger.info(f"🔄 Converting shape {i+1}/{len(shapes)}")
            
            try:
                coordinates = self._convert_single_shape(
                    shape, base_lat, base_lon, scale_info, image_shape
                )
                
                if coordinates:
                    coordinate_set = {
                        'shape_id': i,
                        'coordinates': coordinates,
                        'area_sq_meters': self._calculate_shape_area(coordinates),
                        'perimeter_meters': self._calculate_shape_perimeter(coordinates),
                        'conversion_method': scale_info.get('method', 'estimated'),
                        'confidence': scale_info.get('confidence', 0.5)
                    }
                    coordinate_sets.append(coordinate_set)
                    logger.info(f"✅ Shape {i+1} converted: {len(coordinates)} vertices")
                else:
                    logger.warning(f"⚠️ Failed to convert shape {i+1}")
                    
            except Exception as e:
                logger.error(f"❌ Error converting shape {i+1}: {e}")
                continue
        
        logger.info(f"🎉 Successfully converted {len(coordinate_sets)} shapes")
        return coordinate_sets
    
    def _extract_scale_information(self, text: str, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Extract scale information from text and image analysis"""
        
        scale_info = {
            'method': 'estimated',
            'confidence': 0.3,
            'meters_per_pixel': None,
            'scale_ratio': None,
            'reference_distance': None
        }
        
        # Strategy 1: Extract explicit scale bars and measurements
        scale_bar_info = self._find_scale_bar_measurements(text)
        if scale_bar_info:
            scale_info.update(scale_bar_info)
            scale_info['method'] = 'scale_bar'
            scale_info['confidence'] = 0.9
            return scale_info
        
        # Strategy 2: Extract scale ratios (1:2,257 etc.)
        scale_ratio_info = self._find_scale_ratios(text)
        if scale_ratio_info:
            scale_info.update(scale_ratio_info)
            scale_info['method'] = 'scale_ratio'
            scale_info['confidence'] = 0.8
            return scale_info
        
        # Strategy 3: Use known reference distances (street widths, etc.)
        reference_info = self._estimate_from_references(text, image_shape)
        if reference_info:
            scale_info.update(reference_info)
            scale_info['method'] = 'reference_estimation'
            scale_info['confidence'] = 0.6
            return scale_info
        
        # Strategy 4: Default estimation based on typical parcel map scales
        default_info = self._default_scale_estimation(image_shape)
        scale_info.update(default_info)
        scale_info['method'] = 'default_estimation'
        scale_info['confidence'] = 0.3
        
        return scale_info
    
    def _find_scale_bar_measurements(self, text: str) -> Optional[Dict[str, Any]]:
        """Find scale bar measurements in the text"""
        
        # Patterns for scale bars like "0.01 0.03 0.05 mi" or "0.02 0.04 0.08 km"
        scale_patterns = [
            r'0\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(mi|mile|miles)',
            r'0\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(km|kilometer|kilometers)',
            r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(ft|feet|foot)',
            r'Scale:\s*1\s*inch\s*=\s*([\d.]+)\s*(ft|feet|mile|miles|km)',
        ]
        
        for pattern in scale_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match) == 4:  # Format: 0 val1 val2 val3 unit
                        val1, val2, val3, unit = match
                        # Use the middle value as reference
                        distance = float(val2)
                    else:  # Format: Scale: 1 inch = X unit
                        distance = float(match[0])
                        unit = match[1]
                    
                    # Convert to meters
                    meters = self._convert_to_meters(distance, unit)
                    if meters:
                        logger.info(f"📏 Found scale bar: {distance} {unit} = {meters} meters")
                        return {
                            'reference_distance': meters,
                            'scale_text': f"{distance} {unit}",
                            'meters_per_pixel': meters / 100  # Rough estimate
                        }
                        
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _find_scale_ratios(self, text: str) -> Optional[Dict[str, Any]]:
        """Find scale ratios like 1:2,257"""
        
        ratio_patterns = [
            r'1[:\s]*([,\d]+)',
            r'Scale[:\s]*1[:\s]*([,\d]+)',
            r'([,\d]+)\s*:\s*1',
        ]
        
        for pattern in ratio_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Clean the ratio (remove commas)
                    ratio_str = match.replace(',', '')
                    if ratio_str.isdigit():
                        ratio = int(ratio_str)
                        
                        # Validate reasonable range for map scales
                        if 100 <= ratio <= 100000:
                            logger.info(f"📏 Found scale ratio: 1:{ratio}")
                            
                            # Estimate meters per pixel based on typical map resolutions
                            # Assume 300 DPI for printed maps
                            inches_per_pixel = 1.0 / 300.0
                            real_inches_per_pixel = inches_per_pixel * ratio
                            meters_per_pixel = real_inches_per_pixel * 0.0254
                            
                            return {
                                'scale_ratio': ratio,
                                'meters_per_pixel': meters_per_pixel,
                                'scale_text': f"1:{ratio}"
                            }
                            
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _estimate_from_references(self, text: str, image_shape: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """Estimate scale using known reference objects/distances"""
        
        # Look for lot dimensions or property sizes
        dimension_patterns = [
            r'(\d+)\s*[\'\"]\s*[xX×]\s*(\d+)\s*[\'\"]\s*',  # Feet dimensions
            r'(\d+)\s*ft\s*[xX×]\s*(\d+)\s*ft',
            r'(\d+\.?\d*)\s*acres?',
            r'(\d+)\s*sq\s*ft',
        ]
        
        for pattern in dimension_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    if 'acre' in pattern:
                        acres = float(matches[0])
                        # Typical residential lot is 0.25 acres
                        if 0.1 <= acres <= 5.0:
                            # Estimate based on typical lot proportions
                            sq_meters = acres * 4047  # acres to sq meters
                            side_length = math.sqrt(sq_meters)
                            
                            # Assume the property takes up ~30% of image
                            image_diagonal = math.sqrt(image_shape[0]**2 + image_shape[1]**2)
                            estimated_scale = side_length / (image_diagonal * 0.3)
                            
                            return {
                                'meters_per_pixel': estimated_scale,
                                'reference_type': 'property_acreage',
                                'reference_value': f"{acres} acres"
                            }
                    else:
                        # Dimensional measurements
                        dim1, dim2 = float(matches[0][0]), float(matches[0][1])
                        # Assume these are property dimensions in feet
                        meters1 = dim1 * 0.3048
                        meters2 = dim2 * 0.3048
                        
                        # Rough estimation based on image size
                        avg_dimension = (meters1 + meters2) / 2
                        avg_image_dimension = (image_shape[0] + image_shape[1]) / 2
                        
                        estimated_scale = avg_dimension / (avg_image_dimension * 0.3)
                        
                        return {
                            'meters_per_pixel': estimated_scale,
                            'reference_type': 'property_dimensions',
                            'reference_value': f"{dim1}' x {dim2}'"
                        }
                        
                except (ValueError, IndexError, ZeroDivisionError):
                    continue
        
        return None
    
    def _default_scale_estimation(self, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Default scale estimation for typical parcel maps"""
        
        # Typical residential parcel map scales
        # Assume image shows a property of ~100m x 100m (typical residential lot area)
        typical_property_size = 100  # meters
        
        # Assume property occupies ~40% of the image
        avg_image_dimension = (image_shape[0] + image_shape[1]) / 2
        property_pixels = avg_image_dimension * 0.4
        
        meters_per_pixel = typical_property_size / property_pixels
        
        logger.info(f"📏 Using default scale estimation: {meters_per_pixel:.4f} m/pixel")
        
        return {
            'meters_per_pixel': meters_per_pixel,
            'reference_type': 'default_estimation',
            'reference_value': f"~{typical_property_size}m property"
        }
    
    def _convert_to_meters(self, value: float, unit: str) -> Optional[float]:
        """Convert distance value to meters"""
        
        unit = unit.lower().strip()
        
        conversions = {
            'mi': 1609.34, 'mile': 1609.34, 'miles': 1609.34,
            'km': 1000.0, 'kilometer': 1000.0, 'kilometers': 1000.0,
            'ft': 0.3048, 'feet': 0.3048, 'foot': 0.3048,
            'm': 1.0, 'meter': 1.0, 'meters': 1.0,
            'yd': 0.9144, 'yard': 0.9144, 'yards': 0.9144
        }
        
        return conversions.get(unit, None) and value * conversions[unit]
    
    def _convert_single_shape(self, 
                            shape: np.ndarray, 
                            base_lat: float, 
                            base_lon: float,
                            scale_info: Dict[str, Any],
                            image_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Convert a single shape to geo-coordinates"""
        
        if len(shape) < 3:
            return []
        
        # Get scale
        meters_per_pixel = scale_info.get('meters_per_pixel', 0.5)
        
        # Find shape centroid for relative positioning
        moments = cv2.moments(shape)
        if moments['m00'] == 0:
            return []
        
        centroid_x = int(moments['m10'] / moments['m00'])
        centroid_y = int(moments['m01'] / moments['m00'])
        
        # Image center as reference point
        image_center_x = image_shape[1] // 2
        image_center_y = image_shape[0] // 2
        
        coordinates = []
        
        # Convert each vertex
        for point in shape:
            try:
                pixel_x = int(point[0][0])
                pixel_y = int(point[0][1])
                
                # Calculate offset from image center in pixels
                dx_pixels = pixel_x - image_center_x
                dy_pixels = pixel_y - image_center_y
                
                # Convert to meters (note: Y is inverted in image coordinates)
                dx_meters = dx_pixels * meters_per_pixel
                dy_meters = -dy_pixels * meters_per_pixel  # Negative because Y increases downward
                
                # Calculate new coordinates using geodesic math
                lat, lon = self._offset_coordinates(base_lat, base_lon, dx_meters, dy_meters)
                
                coordinates.append({
                    'latitude': lat,
                    'longitude': lon,
                    'pixel_x': pixel_x,
                    'pixel_y': pixel_y
                })
                
            except (IndexError, ValueError) as e:
                logger.warning(f"Error converting point: {e}")
                continue
        
        return coordinates
    
    def _offset_coordinates(self, 
                          base_lat: float, 
                          base_lon: float, 
                          dx_meters: float, 
                          dy_meters: float) -> Tuple[float, float]:
        """Calculate new coordinates with given offset in meters"""
        
        # Earth's radius in meters
        earth_radius = 6378137.0
        
        # Calculate offsets in degrees
        lat_offset = dy_meters / earth_radius * (180 / math.pi)
        lon_offset = dx_meters / (earth_radius * math.cos(math.radians(base_lat))) * (180 / math.pi)
        
        new_lat = base_lat + lat_offset
        new_lon = base_lon + lon_offset
        
        return new_lat, new_lon
    
    def _calculate_shape_area(self, coordinates: List[Dict[str, Any]]) -> float:
        """Calculate area of shape in square meters using Shoelace formula"""
        
        if len(coordinates) < 3:
            return 0.0
        
        try:
            # Extract lat/lon pairs
            points = [(coord['latitude'], coord['longitude']) for coord in coordinates]
            
            # Use Shoelace formula with geodesic corrections for small areas
            area = 0.0
            n = len(points)
            
            for i in range(n):
                j = (i + 1) % n
                
                # Get points in Web Mercator projection for area calculation
                # Approximation for small areas
                lat1, lon1 = points[i]
                lat2, lon2 = points[j]
                
                # Convert to meters using Web Mercator approximation
                x1 = lon1 * 111320 * math.cos(math.radians(lat1))
                y1 = lat1 * 110540
                x2 = lon2 * 111320 * math.cos(math.radians(lat2))
                y2 = lat2 * 110540
                
                area += x1 * y2 - x2 * y1
            
            return abs(area) / 2.0
            
        except Exception as e:
            logger.warning(f"Error calculating area: {e}")
            return 0.0
    
    def _calculate_shape_perimeter(self, coordinates: List[Dict[str, Any]]) -> float:
        """Calculate perimeter of shape in meters"""
        
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
            
        except Exception as e:
            logger.warning(f"Error calculating perimeter: {e}")
            return 0.0

def create_enhanced_converter() -> EnhancedCoordinateConverter:
    """Factory function to create enhanced converter"""
    return EnhancedCoordinateConverter() 