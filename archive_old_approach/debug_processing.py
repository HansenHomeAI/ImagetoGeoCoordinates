#!/usr/bin/env python3

import os
import sys
import logging
import traceback
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image
import pytesseract
import easyocr
import re
from geopy.geocoders import Nominatim

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

def process_parcel_map_debug():
    """Debug processing of the parcel map with comprehensive logging"""
    
    pdf_file = "LOT 2 324 Dolan Rd Aerial Map.pdf"
    
    logger.info(f"🚀 Starting debug processing of: {pdf_file}")
    
    if not os.path.exists(pdf_file):
        logger.error(f"❌ File not found: {pdf_file}")
        return False
    
    try:
        # Step 1: Convert PDF to image
        logger.info("📄 Step 1: Converting PDF to image...")
        pages = convert_from_path(pdf_file, first_page=1, last_page=1, dpi=300)
        if not pages:
            logger.error("❌ No pages found in PDF")
            return False
        
        pil_image = pages[0]
        logger.info(f"✅ PDF converted - Size: {pil_image.size}")
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        logger.info(f"✅ OpenCV image shape: {opencv_image.shape}")
        
        # Step 2: Text extraction with Tesseract
        logger.info("📖 Step 2: Extracting text with Tesseract...")
        
        # Convert to RGB for Tesseract
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        pil_for_ocr = Image.fromarray(rgb_image)
        
        # Try different Tesseract configurations
        configs = ['--psm 6', '--psm 11', '--psm 12', '--psm 13']
        all_text = []
        
        for config in configs:
            try:
                logger.info(f"🔍 Trying Tesseract config: {config}")
                text = pytesseract.image_to_string(pil_for_ocr, config=config)
                if text.strip():
                    all_text.append(text)
                    logger.info(f"✅ Extracted {len(text)} characters with {config}")
                    logger.debug(f"Sample text: {text[:200]}...")
            except Exception as e:
                logger.error(f"❌ Tesseract {config} failed: {e}")
        
        # Step 3: Text extraction with EasyOCR
        logger.info("🤖 Step 3: Attempting EasyOCR...")
        try:
            reader = easyocr.Reader(['en'], verbose=True)
            results = reader.readtext(opencv_image, detail=1, paragraph=False)
            
            easyocr_texts = []
            for bbox, text, confidence in results:
                if confidence > 0.3:
                    easyocr_texts.append(text)
                    logger.debug(f"📝 EasyOCR: '{text}' (confidence: {confidence:.2f})")
            
            if easyocr_texts:
                all_text.append(" ".join(easyocr_texts))
                logger.info(f"✅ EasyOCR extracted {len(easyocr_texts)} text segments")
            
        except Exception as e:
            logger.error(f"❌ EasyOCR failed: {e}")
            logger.error(traceback.format_exc())
        
        # Combine all text
        combined_text = " ".join(all_text)
        logger.info(f"📊 Total extracted text: {len(combined_text)} characters")
        
        if combined_text:
            logger.info("📄 EXTRACTED TEXT (first 1000 characters):")
            logger.info("=" * 60)
            logger.info(combined_text[:1000])
            logger.info("=" * 60)
        else:
            logger.warning("⚠️ No text extracted from any OCR method")
        
        # Step 4: Enhanced location analysis
        logger.info("🌍 Step 4: Enhanced location analysis...")
        
        # Find street names with improved patterns
        street_patterns = [
            r'(\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:Road|Rd|Street|St|Avenue|Ave|Drive|Dr|Lane|Ln))',
            r'([A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:Road|Rd|Street|St|Avenue|Ave|Drive|Dr|Lane|Ln))'
        ]
        
        streets = []
        for pattern in street_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            streets.extend(matches)
            if matches:
                logger.info(f"🛣️ Found streets with pattern: {matches}")
        
        # Clean up street names
        clean_streets = []
        for street in streets:
            # Remove common OCR artifacts and clean up
            clean_street = re.sub(r'[^\w\s]', ' ', street)
            clean_street = re.sub(r'\s+', ' ', clean_street).strip()
            if len(clean_street) > 5 and 'DOLAN' in clean_street.upper():
                clean_streets.append(clean_street)
        
        logger.info(f"🧹 Cleaned streets: {clean_streets}")
        
        # Find coordinates with enhanced patterns
        coord_patterns = [
            r'(\d{1,3}°\s*\d{1,2}\'\s*[\d.]+\"\s*[NS])',
            r'(\d{1,3}\.\d{4,})',
            r'([-]?\d{1,3}\.\d{4,}\s*,\s*[-]?\d{1,3}\.\d{4,})'
        ]
        
        coordinates = []
        for pattern in coord_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            coordinates.extend(matches)
            if matches:
                logger.info(f"📍 Found coordinates: {matches}")
        
        # Enhanced county/state detection
        county_match = re.search(r'([A-Za-z]+(?:\s+[A-Za-z]+)*\s+County)', combined_text, re.IGNORECASE)
        county = county_match.group(1) if county_match else None
        
        # Better state detection - look for Washington state indicators
        state = None
        if 'Washington' in combined_text:
            state = 'WA'
        elif 'Cowlitz' in combined_text:  # Cowlitz County is in Washington
            state = 'WA'
        else:
            # Try to find two-letter state codes, but filter out common false positives
            state_match = re.search(r'\b(WA|OR|CA|ID|MT|ND|SD|WY|CO|UT|NV|AZ|NM|TX|OK|KS|NE|IA|MN|WI|IL|IN|OH|MI|PA|NY|VT|NH|ME|MA|RI|CT|NJ|DE|MD|VA|WV|KY|TN|NC|SC|GA|FL|AL|MS|LA|AR|MO|AK|HI)\b', combined_text)
            if state_match:
                state = state_match.group(1)
        
        if county:
            logger.info(f"🏛️ Found county: {county}")
        if state:
            logger.info(f"🗺️ Found state: {state}")
        
        # Step 5: Enhanced geocoding with multiple strategies
        if clean_streets or county:
            logger.info("🌐 Step 5: Enhanced geocoding...")
            geolocator = Nominatim(user_agent="ImagetoGeoCoordinates-debug")
            
            geocoding_attempts = []
            
            # Strategy 1: Try the main address
            if clean_streets:
                main_street = clean_streets[0]
                if county and state:
                    geocoding_attempts.append(f"{main_street}, {county}, {state}")
                if county:
                    geocoding_attempts.append(f"{main_street}, {county}")
                geocoding_attempts.append(main_street)
            
            # Strategy 2: Try just "Dolan Road" with location
            if county and state:
                geocoding_attempts.append(f"Dolan Road, {county}, {state}")
            if county:
                geocoding_attempts.append(f"Dolan Road, {county}")
            
            # Strategy 3: Try county + state
            if county and state:
                geocoding_attempts.append(f"{county}, {state}")
            
            for attempt in geocoding_attempts:
                logger.info(f"🔍 Geocoding query: '{attempt}'")
                try:
                    location = geolocator.geocode(attempt, timeout=15)
                    if location:
                        logger.info(f"✅ Geocoding successful!")
                        logger.info(f"📍 Coordinates: {location.latitude}, {location.longitude}")
                        logger.info(f"📍 Address: {location.address}")
                        break
                    else:
                        logger.warning(f"⚠️ No results for query: '{attempt}'")
                except Exception as e:
                    logger.error(f"❌ Geocoding error for '{attempt}': {e}")
        
        # Step 6: Enhanced computer vision analysis
        logger.info("🔍 Step 6: Enhanced computer vision analysis...")
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        logger.info(f"📏 Image dimensions: {gray.shape}")
        
        # Multiple edge detection approaches
        edge_methods = [
            ("Canny_standard", lambda img: cv2.Canny(img, 50, 150)),
            ("Canny_sensitive", lambda img: cv2.Canny(img, 30, 100)),
            ("Canny_aggressive", lambda img: cv2.Canny(img, 100, 200)),
        ]
        
        all_contours = []
        
        for method_name, edge_func in edge_methods:
            logger.info(f"🔍 Trying edge detection: {method_name}")
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply edge detection
            edges = edge_func(blurred)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            logger.info(f"📊 {method_name} found {len(contours)} contours")
            
            # Filter contours by area and shape
            height, width = gray.shape
            min_area = (width * height) * 0.0005  # Reduced minimum area
            max_area = (width * height) * 0.3     # Reduced maximum area
            
            valid_contours = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    # Check if contour is roughly rectangular/polygonal
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if 3 <= len(approx) <= 15:  # Reasonable polygon
                        valid_contours.append((contour, approx, area))
                        logger.debug(f"Valid contour {i}: {len(approx)} vertices, area = {int(area)}")
            
            all_contours.extend(valid_contours)
            logger.info(f"✅ {method_name} contributed {len(valid_contours)} valid contours")
        
        # Remove duplicate contours
        unique_contours = []
        tolerance = 100  # pixels
        
        for contour, approx, area in all_contours:
            is_duplicate = False
            for existing_contour, _, _ in unique_contours:
                # Simple duplicate check based on area and position
                existing_area = cv2.contourArea(existing_contour)
                if abs(area - existing_area) < tolerance * tolerance:
                    # Check if centroids are close
                    M1 = cv2.moments(contour)
                    M2 = cv2.moments(existing_contour)
                    if M1["m00"] > 0 and M2["m00"] > 0:
                        cx1, cy1 = int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"])
                        cx2, cy2 = int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"])
                        distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                        if distance < tolerance:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                unique_contours.append((contour, approx, area))
        
        logger.info(f"🎯 Found {len(unique_contours)} unique property-like contours")
        
        # Analyze the largest contours as potential property boundaries
        if unique_contours:
            # Sort by area (largest first)
            unique_contours.sort(key=lambda x: x[2], reverse=True)
            
            logger.info("🏠 Top property boundary candidates:")
            for i, (contour, approx, area) in enumerate(unique_contours[:5]):
                logger.info(f"  {i+1}. Area: {int(area)}, Vertices: {len(approx)}")
                
                # Calculate approximate coordinates for this shape
                if location:  # If we have a base location
                    vertices = []
                    for point in approx.reshape(-1, 2):
                        x, y = point
                        # Very rough coordinate estimation
                        # This would need proper surveying data for accuracy
                        lat_offset = (y - height/2) * 0.00001  # Rough approximation
                        lon_offset = (x - width/2) * 0.00001   # Rough approximation
                        
                        vertex_lat = location.latitude + lat_offset
                        vertex_lon = location.longitude + lon_offset
                        vertices.append((vertex_lat, vertex_lon))
                    
                    logger.info(f"    Estimated vertices: {vertices}")
        
        logger.info("🎉 Enhanced debug processing completed successfully!")
        
        # Enhanced summary
        logger.info("\n📊 ENHANCED PROCESSING SUMMARY:")
        logger.info(f"✅ Text extracted: {len(combined_text)} characters")
        logger.info(f"🛣️ Streets found: {len(clean_streets)}")
        logger.info(f"📍 Coordinates found: {len(coordinates)}")
        logger.info(f"🏛️ County: {county or 'Not found'}")
        logger.info(f"🗺️ State: {state or 'Not found'}")
        logger.info(f"🔍 Valid contours: {len(unique_contours)}")
        logger.info(f"🌐 Geocoding: {'Success' if 'location' in locals() and location else 'Failed'}")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Debug processing failed: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("🧪 Enhanced Parcel Map Debug Processing")
    print("=" * 40)
    
    success = process_parcel_map_debug()
    
    if success:
        print("\n✅ Enhanced debug processing completed successfully!")
    else:
        print("\n❌ Enhanced debug processing failed!")
    
    sys.exit(0 if success else 1) 