#!/usr/bin/env python3
"""
Advanced Parcel Map Processor - Robust property boundary detection and coordinate calculation
Handles diverse parcel map formats using multiple detection strategies and free/open data sources
"""

import cv2
import numpy as np
import logging
import requests
import json
import re
from typing import List, Dict, Tuple, Optional
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import math
from shapely.geometry import Polygon, Point
from shapely.ops import transform
import pyproj
from functools import partial

logger = logging.getLogger(__name__)

class AdvancedParcelProcessor:
    """Advanced processor for parcel maps with sophisticated boundary detection"""
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="parcel_processor_v2")
        
    def detect_boundaries_multi_strategy(self, image, text_info: Dict) -> List[np.ndarray]:
        """Use multiple strategies to detect property boundaries"""
        logger.info("🎯 Starting multi-strategy boundary detection...")
        
        all_contours = []
        
        # Strategy 1: Classical edge detection with multiple parameters
        edge_contours = self._detect_with_edge_methods(image)
        all_contours.extend(edge_contours)
        logger.info(f"📐 Edge detection found {len(edge_contours)} contours")
        
        # Strategy 2: Color-based segmentation (for property lines)
        color_contours = self._detect_with_color_segmentation(image)
        all_contours.extend(color_contours)
        logger.info(f"🎨 Color segmentation found {len(color_contours)} contours")
        
        # Strategy 3: Morphological operations for line detection
        morph_contours = self._detect_with_morphology(image)
        all_contours.extend(morph_contours)
        logger.info(f"🔧 Morphological ops found {len(morph_contours)} contours")
        
        # Strategy 4: Template matching for common property shapes
        template_contours = self._detect_with_templates(image)
        all_contours.extend(template_contours)
        logger.info(f"📋 Template matching found {len(template_contours)} contours")
        
        # Strategy 5: Scale-based detection using street references
        scale_contours = self._detect_with_scale_reference(image, text_info)
        all_contours.extend(scale_contours)
        logger.info(f"📏 Scale-based detection found {len(scale_contours)} contours")
        
        # Filter and rank all contours
        filtered_contours = self._filter_and_rank_contours(all_contours, image.shape)
        logger.info(f"✅ Final filtered contours: {len(filtered_contours)}")
        
        return filtered_contours
    
    def _detect_with_edge_methods(self, image) -> List[np.ndarray]:
        """Multiple edge detection approaches optimized for parcel maps"""
        contours = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Preprocessing variants
        preprocessed_images = []
        
        # Original grayscale
        preprocessed_images.append(gray)
        
        # Gaussian blur variations
        for kernel_size in [3, 5, 7]:
            blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            preprocessed_images.append(blurred)
        
        # Bilateral filter (preserves edges)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        preprocessed_images.append(bilateral)
        
        # Histogram equalization
        equalized = cv2.equalizeHist(gray)
        preprocessed_images.append(equalized)
        
        # CLAHE (adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_applied = clahe.apply(gray)
        preprocessed_images.append(clahe_applied)
        
        # Edge detection methods
        edge_methods = [
            ("Canny_conservative", lambda img: cv2.Canny(img, 30, 90)),
            ("Canny_standard", lambda img: cv2.Canny(img, 50, 150)),
            ("Canny_aggressive", lambda img: cv2.Canny(img, 100, 200)),
            ("Sobel_combined", self._sobel_combined),
            ("Laplacian", lambda img: cv2.Laplacian(img, cv2.CV_8U)),
        ]
        
        for prep_img in preprocessed_images:
            for method_name, edge_func in edge_methods:
                try:
                    edges = edge_func(prep_img)
                    method_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours.extend(method_contours)
                except Exception as e:
                    logger.debug(f"Edge method {method_name} failed: {e}")
        
        return contours
    
    def _sobel_combined(self, image):
        """Combined Sobel edge detection"""
        sobelx = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=3)
        return cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    
    def _detect_with_color_segmentation(self, image) -> List[np.ndarray]:
        """Detect boundaries using color segmentation"""
        contours = []
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Define ranges for property line colors (often dark lines)
        color_ranges = [
            # Dark lines in HSV
            {"space": hsv, "lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 50])},
            # Red property lines
            {"space": hsv, "lower": np.array([0, 120, 70]), "upper": np.array([10, 255, 255])},
            {"space": hsv, "lower": np.array([170, 120, 70]), "upper": np.array([180, 255, 255])},
            # Blue property lines
            {"space": hsv, "lower": np.array([100, 120, 70]), "upper": np.array([130, 255, 255])},
        ]
        
        for color_range in color_ranges:
            mask = cv2.inRange(color_range["space"], color_range["lower"], color_range["upper"])
            
            # Morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            method_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(method_contours)
        
        return contours
    
    def _detect_with_morphology(self, image) -> List[np.ndarray]:
        """Use morphological operations to detect line structures"""
        contours = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Different kernel sizes and shapes for line detection
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)),  # Horizontal lines
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)),  # Vertical lines
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),   # Square
            cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),  # Cross
        ]
        
        for kernel in kernels:
            # Opening operation to detect lines
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            # Threshold
            _, thresh = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            method_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(method_contours)
        
        return contours
    
    def _detect_with_templates(self, image) -> List[np.ndarray]:
        """Template matching for common property shapes"""
        contours = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create templates for common property shapes
        templates = self._create_property_templates()
        
        for template_name, template in templates.items():
            # Template matching
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.6)  # Threshold for matches
            
            # Convert matches to contours
            for pt in zip(*locations[::-1]):
                h, w = template.shape
                # Create rectangular contour for each match
                rect_contour = np.array([
                    [pt[0], pt[1]], [pt[0] + w, pt[1]], 
                    [pt[0] + w, pt[1] + h], [pt[0], pt[1] + h]
                ]).reshape(-1, 1, 2)
                contours.append(rect_contour)
        
        return contours
    
    def _create_property_templates(self) -> Dict[str, np.ndarray]:
        """Create templates for common property boundary patterns"""
        templates = {}
        
        # Rectangle template
        rect_template = np.zeros((50, 80), dtype=np.uint8)
        cv2.rectangle(rect_template, (5, 5), (75, 45), 255, 2)
        templates["rectangle"] = rect_template
        
        # Square template
        square_template = np.zeros((60, 60), dtype=np.uint8)
        cv2.rectangle(square_template, (5, 5), (55, 55), 255, 2)
        templates["square"] = square_template
        
        # L-shaped template
        l_template = np.zeros((60, 60), dtype=np.uint8)
        cv2.line(l_template, (10, 10), (50, 10), 255, 2)
        cv2.line(l_template, (10, 10), (10, 50), 255, 2)
        templates["l_shape"] = l_template
        
        return templates
    
    def _detect_with_scale_reference(self, image, text_info: Dict) -> List[np.ndarray]:
        """Use scale information from text to guide detection"""
        contours = []
        
        # Extract scale information from text
        scale_info = self._extract_scale_info(text_info.get("extracted_text", ""))
        
        if scale_info:
            # Use scale to determine expected property sizes
            expected_pixel_sizes = self._calculate_expected_sizes(scale_info, image.shape)
            
            # Run targeted detection based on expected sizes
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Adaptive threshold based on scale
            for block_size in expected_pixel_sizes.get("block_sizes", [11, 21, 31]):
                adaptive = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2
                )
                method_contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours.extend(method_contours)
        
        return contours
    
    def _extract_scale_info(self, text: str) -> Optional[Dict]:
        """Extract scale information from map text"""
        scale_patterns = [
            r'1:(\d+,?\d*)',  # Scale ratio like 1:2,257
            r'(\d+\.?\d*)\s*(?:ft|feet|foot)',  # Feet measurements
            r'(\d+\.?\d*)\s*(?:mi|mile|miles)',  # Mile measurements
            r'(\d+\.?\d*)\s*(?:km|kilometer)',  # Kilometer measurements
            r'(\d+\.?\d*)\s*(?:m|meter|metres)',  # Meter measurements
        ]
        
        scale_info = {}
        
        for pattern in scale_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                scale_info["matches"] = matches
                scale_info["pattern"] = pattern
                break
        
        return scale_info if scale_info else None
    
    def _calculate_expected_sizes(self, scale_info: Dict, image_shape: Tuple) -> Dict:
        """Calculate expected property sizes in pixels based on scale"""
        height, width = image_shape[:2]
        
        # Default assumptions for property sizes
        typical_property_acres = [0.25, 0.5, 1.0, 2.0, 5.0]  # Common property sizes
        acre_to_sqft = 43560
        
        expected_sizes = {
            "min_area_pixels": int(width * height * 0.001),  # 0.1% of image
            "max_area_pixels": int(width * height * 0.3),    # 30% of image
            "block_sizes": [11, 21, 31, 41]  # For adaptive threshold
        }
        
        return expected_sizes
    
    def _filter_and_rank_contours(self, contours: List[np.ndarray], image_shape: Tuple) -> List[np.ndarray]:
        """Filter and rank contours by likelihood of being property boundaries"""
        if not contours:
            return []
        
        height, width = image_shape[:2]
        total_area = width * height
        
        # Filtering criteria
        min_area = total_area * 0.0001  # 0.01% of image
        max_area = total_area * 0.4     # 40% of image
        
        candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area < area < max_area:
                # Calculate additional metrics
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    # Compactness (how close to a circle)
                    compactness = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Convex hull ratio
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    # Aspect ratio
                    rect = cv2.minAreaRect(contour)
                    w_rect, h_rect = rect[1]
                    aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect) if min(w_rect, h_rect) > 0 else float('inf')
                    
                    # Calculate score based on property-like characteristics
                    score = self._calculate_property_score(area, compactness, solidity, aspect_ratio)
                    
                    candidates.append({
                        "contour": contour,
                        "score": score,
                        "area": area,
                        "compactness": compactness,
                        "solidity": solidity,
                        "aspect_ratio": aspect_ratio
                    })
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Remove duplicates and return top candidates
        final_contours = self._remove_duplicate_contours([c["contour"] for c in candidates[:20]])
        
        logger.info(f"🎯 Ranked and filtered {len(final_contours)} property candidates")
        return final_contours
    
    def _calculate_property_score(self, area: float, compactness: float, solidity: float, aspect_ratio: float) -> float:
        """Calculate likelihood score that a contour represents a property"""
        score = 0.0
        
        # Prefer moderate compactness (not too circular, not too elongated)
        if 0.1 < compactness < 0.8:
            score += 0.3
        
        # Prefer high solidity (solid shapes)
        if solidity > 0.7:
            score += 0.3
        elif solidity > 0.5:
            score += 0.15
        
        # Prefer reasonable aspect ratios (property-like rectangles)
        if 1.0 < aspect_ratio < 5.0:
            score += 0.4
        elif aspect_ratio < 10.0:
            score += 0.2
        
        return score
    
    def _remove_duplicate_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Remove duplicate or very similar contours"""
        if len(contours) <= 1:
            return contours
        
        unique_contours = []
        tolerance = 50  # pixels
        
        for contour in contours:
            is_duplicate = False
            
            for unique_contour in unique_contours:
                if self._contours_similar(contour, unique_contour, tolerance):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_contours.append(contour)
        
        return unique_contours
    
    def _contours_similar(self, contour1: np.ndarray, contour2: np.ndarray, tolerance: float) -> bool:
        """Check if two contours are similar"""
        try:
            # Compare areas
            area1 = cv2.contourArea(contour1)
            area2 = cv2.contourArea(contour2)
            
            if abs(area1 - area2) > tolerance * tolerance:
                return False
            
            # Compare centroids
            M1 = cv2.moments(contour1)
            M2 = cv2.moments(contour2)
            
            if M1["m00"] > 0 and M2["m00"] > 0:
                cx1, cy1 = int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"])
                cx2, cy2 = int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"])
                centroid_distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                
                return centroid_distance < tolerance
            
            return False
        except:
            return False
    
    def calculate_precise_coordinates(self, contours: List[np.ndarray], location_info: Dict, 
                                   image_shape: Tuple) -> List[List[Dict]]:
        """Calculate precise geographical coordinates using multiple validation methods"""
        logger.info("🌐 Calculating precise geographical coordinates...")
        
        coordinates_list = []
        
        # Get base reference point
        base_coords = self._get_validated_base_coordinates(location_info)
        
        if not base_coords:
            logger.warning("⚠️ No base coordinates found - using fallback methods")
            base_coords = self._estimate_coordinates_from_text(location_info)
        
        # Calculate scale and orientation
        scale_info = self._calculate_map_scale_and_orientation(location_info, image_shape)
        
        for i, contour in enumerate(contours):
            logger.info(f"📍 Processing contour {i+1}/{len(contours)}")
            
            # Convert contour to coordinate points
            contour_coords = self._contour_to_coordinates(
                contour, base_coords, scale_info, image_shape
            )
            
            # Validate coordinates using multiple methods
            validated_coords = self._validate_coordinates(contour_coords, location_info)
            
            if validated_coords:
                coordinates_list.append(validated_coords)
                logger.info(f"✅ Validated coordinates for contour {i+1}")
            else:
                logger.warning(f"⚠️ Could not validate coordinates for contour {i+1}")
        
        return coordinates_list
    
    def _get_validated_base_coordinates(self, location_info: Dict) -> Optional[Dict]:
        """Get and validate base coordinates using multiple sources"""
        
        # Try multiple geocoding strategies
        geocoding_queries = []
        
        # Build specific queries
        if location_info.get("streets") and location_info.get("county") and location_info.get("state"):
            for street in location_info["streets"][:3]:  # Try top 3 streets
                geocoding_queries.append(f"{street}, {location_info['county']}, {location_info['state']}, USA")
        
        if location_info.get("county") and location_info.get("state"):
            geocoding_queries.append(f"{location_info['county']}, {location_info['state']}, USA")
        
        # Try each query
        for query in geocoding_queries:
            try:
                logger.info(f"🔍 Geocoding query: {query}")
                location = self.geolocator.geocode(query, timeout=10)
                
                if location:
                    coords = {
                        "latitude": location.latitude,
                        "longitude": location.longitude,
                        "address": location.address,
                        "query": query,
                        "confidence": "high"
                    }
                    
                    # Validate coordinates are reasonable for US
                    if self._validate_us_coordinates(coords["latitude"], coords["longitude"]):
                        logger.info(f"✅ Validated base coordinates: {coords['latitude']:.6f}, {coords['longitude']:.6f}")
                        return coords
                    
            except Exception as e:
                logger.debug(f"Geocoding failed for '{query}': {e}")
        
        return None
    
    def _validate_us_coordinates(self, lat: float, lon: float) -> bool:
        """Validate that coordinates are within US bounds"""
        # Continental US bounds (approximate)
        return (24.0 <= lat <= 49.0) and (-125.0 <= lon <= -66.0)
    
    def _estimate_coordinates_from_text(self, location_info: Dict) -> Dict:
        """Estimate coordinates when geocoding fails"""
        # State center coordinates as fallback
        state_centers = {
            "WA": {"latitude": 47.7511, "longitude": -120.7401},
            "OR": {"latitude": 44.0582, "longitude": -120.5039},
            "CA": {"latitude": 36.7783, "longitude": -119.4179},
            # Add more states as needed
        }
        
        state = location_info.get("state")
        if state in state_centers:
            coords = state_centers[state].copy()
            coords["address"] = f"Estimated center of {state}"
            coords["confidence"] = "low"
            logger.info(f"📍 Using state center coordinates for {state}")
            return coords
        
        # Default to geographic center of US
        return {
            "latitude": 39.8283,
            "longitude": -98.5795,
            "address": "Geographic center of US (fallback)",
            "confidence": "very_low"
        }
    
    def _calculate_map_scale_and_orientation(self, location_info: Dict, image_shape: Tuple) -> Dict:
        """Calculate map scale and orientation from available information"""
        height, width = image_shape[:2]
        
        # Default scale assumptions
        scale_info = {
            "pixels_per_degree_lat": height / 0.01,  # Assume ~0.01 degree coverage
            "pixels_per_degree_lon": width / 0.01,
            "rotation": 0,  # Assume north is up
            "confidence": "estimated"
        }
        
        # Try to extract scale from text
        text = location_info.get("extracted_text", "")
        
        # Look for scale indicators
        scale_patterns = [
            r'1:(\d+,?\d*)',  # Scale ratio
            r'(\d+\.?\d*)\s*mi',  # Miles
            r'(\d+\.?\d*)\s*km',  # Kilometers
            r'(\d+\.?\d*)\s*ft',  # Feet
        ]
        
        for pattern in scale_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    scale_value = float(matches[0].replace(',', ''))
                    # Adjust scale calculation based on units
                    if 'mi' in pattern:
                        # Miles to degrees (approximate)
                        degrees_covered = scale_value / 69.0  # ~69 miles per degree
                        scale_info["pixels_per_degree_lat"] = height / degrees_covered
                        scale_info["pixels_per_degree_lon"] = width / degrees_covered
                        scale_info["confidence"] = "calculated"
                    elif ':' in pattern:
                        # Scale ratio - estimate coverage
                        estimated_coverage = 0.01 * (scale_value / 2000)  # Rough estimate
                        scale_info["pixels_per_degree_lat"] = height / estimated_coverage
                        scale_info["pixels_per_degree_lon"] = width / estimated_coverage
                        scale_info["confidence"] = "derived"
                    break
                except:
                    continue
        
        logger.info(f"📏 Scale calculation: {scale_info['confidence']} confidence")
        return scale_info
    
    def _contour_to_coordinates(self, contour: np.ndarray, base_coords: Dict, 
                              scale_info: Dict, image_shape: Tuple) -> List[Dict]:
        """Convert contour pixels to geographical coordinates"""
        height, width = image_shape[:2]
        center_x, center_y = width / 2, height / 2
        
        base_lat = base_coords["latitude"]
        base_lon = base_coords["longitude"]
        
        coordinates = []
        
        for point in contour.reshape(-1, 2):
            x, y = point
            
            # Convert to relative position from center
            rel_x = (x - center_x) / scale_info["pixels_per_degree_lon"]
            rel_y = (center_y - y) / scale_info["pixels_per_degree_lat"]  # Y is flipped
            
            # Calculate final coordinates
            latitude = base_lat + rel_y
            longitude = base_lon + rel_x
            
            coordinates.append({
                "latitude": round(latitude, 8),
                "longitude": round(longitude, 8),
                "pixel_x": int(x),
                "pixel_y": int(y)
            })
        
        return coordinates
    
    def _validate_coordinates(self, coordinates: List[Dict], location_info: Dict) -> Optional[List[Dict]]:
        """Validate coordinates using multiple methods"""
        
        if not coordinates:
            return None
        
        # Basic validation - check if coordinates are reasonable
        for coord in coordinates:
            if not self._validate_us_coordinates(coord["latitude"], coord["longitude"]):
                logger.warning(f"❌ Invalid coordinates: {coord['latitude']}, {coord['longitude']}")
                return None
        
        # Check if coordinates form a reasonable property shape
        if len(coordinates) >= 3:
            try:
                # Create polygon and check if it's valid
                coords_tuples = [(c["longitude"], c["latitude"]) for c in coordinates]
                polygon = Polygon(coords_tuples)
                
                if not polygon.is_valid:
                    logger.warning("❌ Invalid polygon shape")
                    return None
                
                # Check area is reasonable for a property
                area_sq_degrees = polygon.area
                # Convert to approximate acres (very rough)
                area_acres = area_sq_degrees * 69 * 69 * 640  # Very rough conversion
                
                if 0.01 < area_acres < 10000:  # Reasonable property size range
                    logger.info(f"✅ Property area: ~{area_acres:.2f} acres")
                    return coordinates
                else:
                    logger.warning(f"❌ Unreasonable property area: {area_acres:.2f} acres")
                    
            except Exception as e:
                logger.warning(f"❌ Polygon validation failed: {e}")
        
        return coordinates  # Return even if validation is inconclusive 