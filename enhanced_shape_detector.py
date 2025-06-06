#!/usr/bin/env python3
"""
Enhanced Shape Detection for Parcel Maps
Robust property boundary detection with adaptive filtering and multiple detection strategies
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

class EnhancedShapeDetector:
    """Enhanced shape detection with adaptive filtering for parcel maps"""
    
    def __init__(self):
        self.debug_mode = True
        
    def detect_property_boundaries(self, image: np.ndarray) -> List[np.ndarray]:
        """Enhanced property boundary detection with multiple strategies"""
        
        logger.info("🔍 Starting enhanced property boundary detection...")
        
        if image is None or image.size == 0:
            logger.error("Invalid input image")
            return []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply multiple detection strategies
        all_contours = []
        
        # Strategy 1: Adaptive threshold with morphology
        contours_1 = self._detect_with_adaptive_morphology(gray)
        logger.info(f"📊 Strategy 1 (Adaptive+Morphology): {len(contours_1)} contours")
        all_contours.extend(contours_1)
        
        # Strategy 2: Multi-scale Canny edge detection
        contours_2 = self._detect_with_multiscale_canny(gray)
        logger.info(f"📊 Strategy 2 (Multi-scale Canny): {len(contours_2)} contours")
        all_contours.extend(contours_2)
        
        # Strategy 3: Blob detection for enclosed areas
        contours_3 = self._detect_with_blob_analysis(gray)
        logger.info(f"📊 Strategy 3 (Blob analysis): {len(contours_3)} contours")
        all_contours.extend(contours_3)
        
        # Strategy 4: Line detection and polygon formation
        contours_4 = self._detect_with_line_analysis(gray)
        logger.info(f"📊 Strategy 4 (Line analysis): {len(contours_4)} contours")
        all_contours.extend(contours_4)
        
        logger.info(f"🔢 Total contours from all strategies: {len(all_contours)}")
        
        # Enhanced filtering with relaxed criteria
        filtered_contours = self._enhanced_filtering(all_contours, gray.shape)
        
        # Deduplicate and rank by quality
        final_contours = self._deduplicate_and_rank(filtered_contours, gray.shape)
        
        logger.info(f"✅ Final filtered contours: {len(final_contours)}")
        
        return final_contours
    
    def _detect_with_adaptive_morphology(self, gray: np.ndarray) -> List[np.ndarray]:
        """Detect contours using adaptive threshold and morphological operations"""
        
        contours = []
        
        # Multiple adaptive threshold parameters
        block_sizes = [15, 25, 35, 45]
        c_values = [5, 10, 15, 20]
        
        for block_size in block_sizes:
            for c_val in c_values:
                try:
                    # Adaptive threshold
                    thresh = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY, block_size, c_val
                    )
                    
                    # Morphological operations to clean up
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    
                    # Find contours
                    contours_found, _ = cv2.findContours(
                        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    contours.extend(contours_found)
                    
                except Exception as e:
                    logger.warning(f"Adaptive threshold failed: {e}")
                    continue
        
        return contours
    
    def _detect_with_multiscale_canny(self, gray: np.ndarray) -> List[np.ndarray]:
        """Multi-scale Canny edge detection with different parameters"""
        
        contours = []
        
        # Blur variations for different scales
        blur_kernels = [(3, 3), (5, 5), (7, 7)]
        
        # Canny parameters for different sensitivities
        canny_params = [
            (30, 100),   # Conservative
            (50, 150),   # Standard
            (20, 80),    # Sensitive
            (100, 200),  # Aggressive
        ]
        
        for blur_kernel in blur_kernels:
            for low_thresh, high_thresh in canny_params:
                try:
                    # Apply Gaussian blur
                    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
                    
                    # Canny edge detection
                    edges = cv2.Canny(blurred, low_thresh, high_thresh)
                    
                    # Dilate edges to close gaps
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    edges = cv2.dilate(edges, kernel, iterations=1)
                    
                    # Find contours
                    contours_found, _ = cv2.findContours(
                        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    contours.extend(contours_found)
                    
                except Exception as e:
                    logger.warning(f"Canny detection failed: {e}")
                    continue
        
        return contours
    
    def _detect_with_blob_analysis(self, gray: np.ndarray) -> List[np.ndarray]:
        """Detect enclosed areas using blob analysis"""
        
        contours = []
        
        try:
            # Otsu thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if needed (we want white areas as foreground)
            if np.mean(thresh) > 127:
                thresh = cv2.bitwise_not(thresh)
            
            # Fill holes in the binary image
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours in the filled image
            contours_found, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            contours.extend(contours_found)
            
        except Exception as e:
            logger.warning(f"Blob analysis failed: {e}")
        
        return contours
    
    def _detect_with_line_analysis(self, gray: np.ndarray) -> List[np.ndarray]:
        """Detect lines and form polygons from intersections"""
        
        contours = []
        
        try:
            # Edge detection for line detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Hough line detection
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, threshold=50, 
                minLineLength=30, maxLineGap=10
            )
            
            if lines is not None and len(lines) > 4:
                # Create a mask with the detected lines
                line_mask = np.zeros_like(gray)
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
                
                # Find contours in the line mask
                contours_found, _ = cv2.findContours(
                    line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                contours.extend(contours_found)
        
        except Exception as e:
            logger.warning(f"Line analysis failed: {e}")
        
        return contours
    
    def _enhanced_filtering(self, contours: List[np.ndarray], image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """Enhanced filtering with relaxed criteria for property boundaries"""
        
        if not contours:
            return []
        
        height, width = image_shape
        image_area = height * width
        min_area = image_area * 0.0001  # Much lower minimum (was 0.001)
        max_area = image_area * 0.8      # Allow larger areas
        
        filtered = []
        
        for contour in contours:
            try:
                # Basic area filter
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    continue
                
                # Perimeter filter
                perimeter = cv2.arcLength(contour, True)
                if perimeter < 20:  # Much lower minimum (was 100)
                    continue
                
                # Aspect ratio and shape analysis
                if self._is_potential_property_shape(contour, image_shape):
                    filtered.append(contour)
                    
            except Exception as e:
                logger.debug(f"Error filtering contour: {e}")
                continue
        
        logger.info(f"🔍 Enhanced filtering: {len(contours)} -> {len(filtered)} contours")
        return filtered
    
    def _is_potential_property_shape(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> bool:
        """Check if contour could be a property boundary with relaxed criteria"""
        
        try:
            # Approximate polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)  # More tolerant approximation
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Property boundaries typically have 3-20 vertices (was 4-12)
            if len(approx) < 3 or len(approx) > 20:
                return False
            
            # Bounding rectangle analysis
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's not too thin (lines) or too square (dots)
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if aspect_ratio > 20:  # Not a thin line
                return False
            
            # Check if it occupies reasonable space
            rect_area = w * h
            contour_area = cv2.contourArea(contour)
            
            if rect_area > 0:
                fill_ratio = contour_area / rect_area
                if fill_ratio < 0.1 or fill_ratio > 0.95:  # More tolerant range
                    return False
            
            # Check if it's not at the very edge of the image
            height, width = image_shape
            margin = 10
            
            if (x < margin or y < margin or 
                x + w > width - margin or y + h > height - margin):
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error in shape analysis: {e}")
            return False
    
    def _deduplicate_and_rank(self, contours: List[np.ndarray], image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """Remove duplicates and rank contours by quality"""
        
        if not contours:
            return []
        
        # Calculate quality scores for all contours
        contour_data = []
        
        for i, contour in enumerate(contours):
            try:
                quality_score = self._calculate_contour_quality(contour, image_shape)
                contour_data.append((contour, quality_score, i))
            except Exception as e:
                logger.debug(f"Error calculating quality for contour {i}: {e}")
                continue
        
        # Sort by quality score (highest first)
        contour_data.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates using similarity comparison
        unique_contours = []
        
        for contour, quality, _ in contour_data:
            is_duplicate = False
            
            for existing_contour in unique_contours:
                if self._contours_similar(contour, existing_contour):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_contours.append(contour)
                
                # Limit to top candidates
                if len(unique_contours) >= 10:
                    break
        
        logger.info(f"🎯 Deduplication: {len(contours)} -> {len(unique_contours)} unique contours")
        return unique_contours
    
    def _calculate_contour_quality(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """Calculate quality score for a contour"""
        
        try:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                return 0.0
            
            # Compactness (4π * area / perimeter²)
            compactness = (4 * np.pi * area) / (perimeter * perimeter)
            
            # Polygon approximation quality
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            simplicity = 1.0 / max(len(approx), 1)
            
            # Size relative to image
            image_area = image_shape[0] * image_shape[1]
            size_score = min(area / (image_area * 0.1), 1.0)
            
            # Convexity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / max(hull_area, 1)
            
            # Combined quality score
            quality = (compactness * 0.3 + 
                      simplicity * 0.2 + 
                      size_score * 0.3 + 
                      convexity * 0.2)
            
            return quality
            
        except Exception as e:
            logger.debug(f"Error calculating quality: {e}")
            return 0.0
    
    def _contours_similar(self, contour1: np.ndarray, contour2: np.ndarray, threshold: float = 0.8) -> bool:
        """Check if two contours are similar (for deduplication)"""
        
        try:
            # Compare using Hu moments
            moments1 = cv2.moments(contour1)
            moments2 = cv2.moments(contour2)
            
            if moments1['m00'] == 0 or moments2['m00'] == 0:
                return False
            
            hu1 = cv2.HuMoments(moments1).flatten()
            hu2 = cv2.HuMoments(moments2).flatten()
            
            # Calculate similarity using correlation coefficient
            correlation = np.corrcoef(hu1, hu2)[0, 1]
            
            if np.isnan(correlation):
                return False
            
            return abs(correlation) > threshold
            
        except Exception as e:
            logger.debug(f"Error comparing contours: {e}")
            return False

def create_enhanced_detector() -> EnhancedShapeDetector:
    """Factory function to create enhanced detector"""
    return EnhancedShapeDetector() 