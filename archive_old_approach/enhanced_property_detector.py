#!/usr/bin/env python3
"""
Enhanced Property Detector - Distinguishes between property boundaries and map annotations
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict
import math

logger = logging.getLogger(__name__)

class EnhancedPropertyDetector:
    """Enhanced detector that filters out text/annotations and focuses on property boundaries"""
    
    def __init__(self):
        self.min_property_area_sqm = 100  # Minimum 100 sq meters (about 0.025 acres)
        self.max_property_area_sqm = 40000  # Maximum 4 hectares (about 10 acres)
        
    def detect_property_boundaries(self, image, text_info: Dict = None) -> List[np.ndarray]:
        """Detect actual property boundaries, filtering out text and annotations"""
        logger.info("🏠 Starting enhanced property boundary detection...")
        
        # Step 1: Detect all potential contours
        all_contours = self._detect_all_contours(image)
        logger.info(f"📊 Found {len(all_contours)} total contours")
        
        # Step 2: Filter out text and small annotations
        non_text_contours = self._filter_text_elements(all_contours, image)
        logger.info(f"📝 After text filtering: {len(non_text_contours)} contours")
        
        # Step 3: Filter by property-like characteristics
        property_candidates = self._filter_property_characteristics(non_text_contours, image.shape)
        logger.info(f"🏠 Property candidates: {len(property_candidates)} contours")
        
        # Step 4: Use context clues from text to validate
        if text_info:
            validated_properties = self._validate_with_context(property_candidates, text_info, image.shape)
            logger.info(f"✅ Context-validated properties: {len(validated_properties)} contours")
            return validated_properties
        
        return property_candidates
    
    def _detect_all_contours(self, image) -> List[np.ndarray]:
        """Detect all contours using multiple methods"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        all_contours = []
        
        # Method 1: Multiple Canny edge detection
        canny_params = [
            (10, 50),   # Very sensitive
            (30, 90),   # Conservative  
            (50, 150),  # Standard
            (100, 200), # Aggressive
        ]
        
        for low, high in canny_params:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, low, high)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
        
        # Method 2: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)
        
        # Method 3: Color-based detection for property lines
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Dark lines (common for property boundaries)
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 100]))
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)
        
        return all_contours
    
    def _filter_text_elements(self, contours: List[np.ndarray], image) -> List[np.ndarray]:
        """Filter out text elements and small annotations"""
        height, width = image.shape[:2]
        filtered_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip very small contours (likely text or noise)
            if area < 500:  # Minimum area threshold
                continue
            
            # Analyze aspect ratio to filter out text-like shapes
            rect = cv2.minAreaRect(contour)
            w, h = rect[1]
            if min(w, h) > 0:
                aspect_ratio = max(w, h) / min(w, h)
                
                # Skip very elongated shapes (likely text lines)
                if aspect_ratio > 10:
                    continue
            
            # Check if contour is too close to image edges (likely map border elements)
            x, y, w, h = cv2.boundingRect(contour)
            margin = 50
            if (x < margin or y < margin or 
                x + w > width - margin or y + h > height - margin):
                continue
            
            # Analyze contour complexity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                # Circularity test - text tends to be less circular
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Skip very non-circular shapes (likely text)
                if circularity < 0.1:
                    continue
            
            filtered_contours.append(contour)
        
        return filtered_contours
    
    def _filter_property_characteristics(self, contours: List[np.ndarray], image_shape: Tuple) -> List[np.ndarray]:
        """Filter contours based on property-like characteristics"""
        height, width = image_shape[:2]
        image_area = width * height
        
        property_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Property size filtering (more realistic thresholds)
            min_area = image_area * 0.001   # 0.1% of image
            max_area = image_area * 0.2     # 20% of image
            
            if not (min_area <= area <= max_area):
                continue
            
            # Analyze shape characteristics
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Compactness (how close to a circle)
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            
            # Solidity (how solid the shape is)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Approximate polygon to count vertices
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = len(approx)
            
            # Property-like characteristics:
            # - Reasonable compactness (not too elongated)
            # - High solidity (solid shape)
            # - Reasonable number of vertices (3-20 for most properties)
            if (0.1 <= compactness <= 1.0 and 
                solidity >= 0.7 and 
                3 <= vertices <= 20):
                
                property_contours.append(contour)
                logger.debug(f"✅ Property candidate: area={area:.0f}, vertices={vertices}, compactness={compactness:.3f}")
        
        return property_contours
    
    def _validate_with_context(self, contours: List[np.ndarray], text_info: Dict, image_shape: Tuple) -> List[np.ndarray]:
        """Use text context to validate property boundaries"""
        validated_contours = []
        
        # Extract useful context
        extracted_text = text_info.get('extracted_text', '')
        
        # Look for scale information
        scale_info = self._extract_scale_from_text(extracted_text)
        
        # Look for property size hints
        size_hints = self._extract_size_hints(extracted_text)
        
        for contour in contours:
            area_pixels = cv2.contourArea(contour)
            
            # Estimate real-world area if we have scale information
            if scale_info:
                estimated_area_sqm = self._estimate_real_area(area_pixels, scale_info, image_shape)
                
                # Check if estimated area is reasonable for a property
                if self.min_property_area_sqm <= estimated_area_sqm <= self.max_property_area_sqm:
                    validated_contours.append(contour)
                    logger.info(f"✅ Validated property: ~{estimated_area_sqm:.0f} sq meters")
                else:
                    logger.debug(f"❌ Rejected: estimated area {estimated_area_sqm:.0f} sq meters")
            else:
                # Without scale info, use relative size validation
                height, width = image_shape[:2]
                relative_area = area_pixels / (width * height)
                
                # Properties should be a reasonable fraction of the map
                if 0.005 <= relative_area <= 0.15:  # 0.5% to 15% of image
                    validated_contours.append(contour)
                    logger.info(f"✅ Validated property: {relative_area*100:.2f}% of image area")
        
        return validated_contours
    
    def _extract_scale_from_text(self, text: str) -> Dict:
        """Extract scale information from text"""
        import re
        
        scale_patterns = [
            r'1:(\d+,?\d*)',  # Scale ratio like 1:2,257
            r'(\d+\.?\d*)\s*(?:ft|feet)',  # Feet measurements
            r'(\d+\.?\d*)\s*(?:mi|mile)',  # Mile measurements
        ]
        
        for pattern in scale_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return {"type": "scale", "value": matches[0], "pattern": pattern}
        
        return {}
    
    def _extract_size_hints(self, text: str) -> List[str]:
        """Extract property size hints from text"""
        import re
        
        size_patterns = [
            r'(\d+\.?\d*)\s*(?:acre|acres)',
            r'(\d+\.?\d*)\s*(?:sq\s*ft|square\s*feet)',
            r'(\d+\.?\d*)\s*(?:hectare|hectares)',
        ]
        
        hints = []
        for pattern in size_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            hints.extend(matches)
        
        return hints
    
    def _estimate_real_area(self, area_pixels: float, scale_info: Dict, image_shape: Tuple) -> float:
        """Estimate real-world area from pixel area"""
        # This is a simplified estimation - in practice would need more sophisticated scale calculation
        height, width = image_shape[:2]
        
        # Default assumption: image covers roughly 1 km x 1 km
        default_coverage_m = 1000
        pixels_per_meter = width / default_coverage_m
        
        # Convert pixel area to square meters
        area_sqm = area_pixels / (pixels_per_meter ** 2)
        
        return area_sqm

def integrate_enhanced_detector():
    """Integration function to add enhanced detection to main app"""
    logger.info("🔧 Enhanced property detector ready for integration")
    return EnhancedPropertyDetector()

if __name__ == "__main__":
    # Test the enhanced detector
    detector = EnhancedPropertyDetector()
    logger.info("✅ Enhanced Property Detector initialized successfully") 