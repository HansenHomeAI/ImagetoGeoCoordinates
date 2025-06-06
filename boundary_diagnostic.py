#!/usr/bin/env python3
"""
Simple Boundary Diagnostic Tool
"""

import cv2
import numpy as np
from pdf2image import convert_from_path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_parcel_map():
    """Load the parcel map"""
    logger.info("📄 Loading parcel map...")
    
    try:
        pages = convert_from_path("LOT 2 324 Dolan Rd Aerial Map.pdf", first_page=1, last_page=1, dpi=300)
        if pages:
            pil_image = pages[0]
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            logger.info(f"✅ Image loaded - Shape: {image.shape}")
            return image
        else:
            logger.error("❌ No pages found")
            return None
    except Exception as e:
        logger.error(f"💥 Error: {e}")
        return None

def analyze_boundaries(image):
    """Analyze boundary detection on the image"""
    logger.info("🔍 Analyzing boundary detection...")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logger.info(f"📊 Image stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
    
    # Test different Canny parameters
    edge_tests = [
        (10, 50, "Very sensitive"),
        (30, 90, "Conservative"),
        (50, 150, "Standard"),
        (100, 200, "Aggressive"),
        (200, 300, "Very aggressive")
    ]
    
    best_result = None
    best_count = 0
    
    for low, high, name in edge_tests:
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, low, high)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        height, width = gray.shape
        min_area = (width * height) * 0.0001  # 0.01% of image
        max_area = (width * height) * 0.3     # 30% of image
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                valid_contours.append(contour)
        
        logger.info(f"🔍 {name} (Canny {low}-{high}): {len(contours)} total, {len(valid_contours)} valid contours")
        
        if len(valid_contours) > best_count:
            best_count = len(valid_contours)
            best_result = (name, low, high, valid_contours, edges)
    
    if best_result:
        name, low, high, contours, edges = best_result
        logger.info(f"🎯 Best result: {name} with {len(contours)} valid contours")
        
        # Save diagnostic images
        cv2.imwrite("debug_original.png", image)
        cv2.imwrite("debug_grayscale.png", gray)
        cv2.imwrite("debug_edges.png", edges)
        
        # Draw contours on original image
        contour_image = image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
        cv2.imwrite("debug_contours.png", contour_image)
        
        logger.info("💾 Saved debug images: debug_*.png")
        
        # Analyze contour properties
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            logger.info(f"📏 Contour areas: min={min(areas):.0f}, max={max(areas):.0f}, mean={np.mean(areas):.0f}")
            
            for i, contour in enumerate(contours[:5]):  # Analyze first 5 contours
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                vertices = len(cv2.approxPolyDP(contour, 0.02 * perimeter, True))
                logger.info(f"📐 Contour {i+1}: area={area:.0f}, perimeter={perimeter:.0f}, vertices={vertices}")
    else:
        logger.warning("⚠️ No valid contours found with any method")
    
    return best_count > 0

def main():
    """Run the diagnostic"""
    logger.info("🚀 Starting boundary diagnostic...")
    
    image = load_parcel_map()
    if image is None:
        logger.error("❌ Failed to load image")
        return
    
    success = analyze_boundaries(image)
    
    if success:
        logger.info("✅ Boundary detection appears possible - check debug images")
    else:
        logger.error("❌ No boundaries detected - image may need different approach")

if __name__ == "__main__":
    main() 