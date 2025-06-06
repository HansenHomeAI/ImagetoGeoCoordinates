#!/usr/bin/env python3
"""
Enhanced Debug Processing for Parcel Maps
Tests the improved parcel map processing system with enhanced modules
"""

import os
import cv2
import numpy as np
import logging
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import easyocr
import json
from datetime import datetime

# Import our enhanced modules
from enhanced_shape_detector import create_enhanced_detector
from enhanced_geocoder import create_enhanced_geocoder
from enhanced_coordinate_converter import create_enhanced_converter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

def process_parcel_map_enhanced(file_path: str):
    """Enhanced parcel map processing with improved modules"""
    
    print("🚀 Enhanced Parcel Map Debug Processing")
    print("=" * 50)
    
    logger.info(f"🚀 Starting enhanced processing of: {os.path.basename(file_path)}")
    
    try:
        # Step 1: Load image
        logger.info("📄 Step 1: Loading and preprocessing image...")
        image = load_image_enhanced(file_path)
        if image is None:
            logger.error("❌ Failed to load image")
            return None
        
        logger.info(f"✅ Image loaded - Shape: {image.shape}")
        
        # Step 2: Enhanced text extraction
        logger.info("📖 Step 2: Enhanced text extraction...")
        extracted_text = extract_text_enhanced(image)
        logger.info(f"📊 Total extracted text: {len(extracted_text)} characters")
        
        if len(extracted_text) > 0:
            logger.info("📄 EXTRACTED TEXT (first 1000 characters):")
            logger.info("=" * 60)
            logger.info(extracted_text[:1000])
            logger.info("=" * 60)
        
        # Step 3: Enhanced location analysis
        logger.info("🌍 Step 3: Enhanced location analysis...")
        geocoder = create_enhanced_geocoder()
        location_info = geocoder.analyze_location_context(extracted_text)
        
        logger.info(f"📍 Location Analysis Results:")
        logger.info(f"   • Primary address: {location_info.get('primary_address')}")
        logger.info(f"   • Streets found: {len(location_info.get('streets', []))}")
        logger.info(f"   • County: {location_info.get('county')}")
        logger.info(f"   • State: {location_info.get('state')}")
        logger.info(f"   • Confidence: {location_info.get('confidence_score', 0):.2f}")
        
        # Step 4: Enhanced geocoding
        logger.info("🌐 Step 4: Enhanced geocoding...")
        base_location = geocoder.geocode_with_context(location_info)
        
        if base_location:
            logger.info(f"✅ Geocoding successful!")
            logger.info(f"   📍 Coordinates: {base_location['latitude']:.6f}, {base_location['longitude']:.6f}")
            logger.info(f"   📍 Address: {base_location['address']}")
            logger.info(f"   📍 Confidence: {base_location.get('confidence', 0):.2f}")
            logger.info(f"   📍 Source: {base_location.get('source', 'unknown')}")
        else:
            logger.warning("⚠️ Geocoding failed")
            return None
        
        # Step 5: Enhanced shape detection
        logger.info("🔍 Step 5: Enhanced shape detection...")
        shape_detector = create_enhanced_detector()
        detected_shapes = shape_detector.detect_property_boundaries(image)
        
        logger.info(f"🏠 Shape Detection Results:")
        logger.info(f"   • Shapes detected: {len(detected_shapes)}")
        
        if len(detected_shapes) == 0:
            logger.warning("⚠️ No valid shapes detected")
            return {
                'success': False,
                'text_extracted': len(extracted_text),
                'location_info': location_info,
                'base_location': base_location,
                'shapes_detected': 0,
                'coordinates_generated': 0
            }
        
        # Step 6: Enhanced coordinate conversion
        logger.info("📐 Step 6: Enhanced coordinate conversion...")
        coordinate_converter = create_enhanced_converter()
        coordinate_sets = coordinate_converter.convert_shapes_to_coordinates(
            detected_shapes, base_location, image.shape, extracted_text
        )
        
        logger.info(f"🎯 Coordinate Conversion Results:")
        logger.info(f"   • Coordinate sets generated: {len(coordinate_sets)}")
        
        for i, coord_set in enumerate(coordinate_sets):
            logger.info(f"   • Shape {i+1}:")
            logger.info(f"     - Vertices: {len(coord_set['coordinates'])}")
            logger.info(f"     - Area: {coord_set['area_sq_meters']:.1f} sq meters")
            logger.info(f"     - Perimeter: {coord_set['perimeter_meters']:.1f} meters")
            logger.info(f"     - Method: {coord_set['conversion_method']}")
            logger.info(f"     - Confidence: {coord_set['confidence']:.2f}")
        
        # Step 7: Generate output
        result = {
            'success': True,
            'processing_timestamp': datetime.now().isoformat(),
            'image_shape': image.shape,
            'text_analysis': {
                'extracted_text_length': len(extracted_text),
                'location_info': location_info
            },
            'geocoding': base_location,
            'shape_detection': {
                'shapes_detected': len(detected_shapes),
                'detection_method': 'enhanced_multi_strategy'
            },
            'coordinate_conversion': {
                'coordinate_sets': coordinate_sets,
                'total_shapes_converted': len(coordinate_sets)
            }
        }
        
        # Save detailed results
        output_file = 'enhanced_processing_results.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"💾 Detailed results saved to: {output_file}")
        
        logger.info("🎉 Enhanced processing completed successfully!")
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 ENHANCED PROCESSING SUMMARY:")
        print("=" * 50)
        print(f"✅ Text extracted: {len(extracted_text)} characters")
        print(f"🛣️ Streets found: {len(location_info.get('streets', []))}")
        print(f"🏛️ County: {location_info.get('county', 'Not found')}")
        print(f"🗺️ State: {location_info.get('state', 'Not found')}")
        print(f"📍 Geocoding: {'Success' if base_location else 'Failed'}")
        print(f"🔍 Shapes detected: {len(detected_shapes)}")
        print(f"🌍 Coordinates generated: {len(coordinate_sets)}")
        
        if coordinate_sets:
            print(f"\n🎯 COORDINATE DETAILS:")
            for i, coord_set in enumerate(coordinate_sets):
                print(f"   Shape {i+1}: {len(coord_set['coordinates'])} vertices, "
                      f"{coord_set['area_sq_meters']:.1f} m² area")
        
        return result
        
    except Exception as e:
        logger.error(f"💥 Enhanced processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def load_image_enhanced(file_path: str) -> np.ndarray:
    """Enhanced image loading with better error handling"""
    
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        logger.info(f"📁 Loading file type: {file_ext}")
        
        if file_ext == '.pdf':
            logger.info("📄 Converting PDF to image...")
            pages = convert_from_path(file_path, first_page=1, last_page=1, dpi=300)
            if pages:
                pil_image = pages[0]
                logger.info(f"✅ PDF converted - Size: {pil_image.size}")
                opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                return opencv_image
            else:
                logger.error("❌ No pages found in PDF")
                return None
        else:
            logger.info(f"🖼️ Loading image format: {file_ext}")
            pil_image = Image.open(file_path)
            logger.info(f"📏 Original image mode: {pil_image.mode}, size: {pil_image.size}")
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                logger.info("🔄 Converted image to RGB")
            
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            logger.info(f"✅ Image loaded successfully - OpenCV shape: {opencv_image.shape}")
            return opencv_image
            
    except Exception as e:
        logger.error(f"💥 Error loading image: {e}")
        return None

def extract_text_enhanced(image: np.ndarray) -> str:
    """Enhanced text extraction using multiple OCR engines"""
    
    logger.info("📖 Starting enhanced text extraction...")
    all_text = []
    
    # Try Tesseract with multiple configurations
    tesseract_configs = [
        '--psm 6',
        '--psm 11',
        '--psm 12',
        '--psm 13'
    ]
    
    for config in tesseract_configs:
        try:
            logger.info(f"🔍 Trying Tesseract config: {config}")
            text = pytesseract.image_to_string(image, config=config)
            if text.strip():
                all_text.append(text.strip())
                logger.info(f"✅ Extracted {len(text)} characters with {config}")
                logger.debug(f"Sample text: {text[:200]}...")
        except Exception as e:
            logger.warning(f"Tesseract config {config} failed: {e}")
            continue
    
    # Try EasyOCR
    try:
        logger.info("🤖 Attempting EasyOCR...")
        reader = easyocr.Reader(['en'], verbose=False)
        results = reader.readtext(image, detail=1, paragraph=False)
        
        easyocr_text = []
        for bbox, text, confidence in results:
            if confidence > 0.3:  # Lower threshold for more text
                easyocr_text.append(text)
                logger.debug(f"📝 EasyOCR: '{text}' (confidence: {confidence:.2f})")
        
        if easyocr_text:
            combined_easyocr = ' '.join(easyocr_text)
            all_text.append(combined_easyocr)
            logger.info(f"✅ EasyOCR extracted {len(easyocr_text)} text segments")
    
    except Exception as e:
        logger.warning(f"EasyOCR failed: {e}")
    
    # Combine all text
    if all_text:
        # Use the longest text extraction as primary
        all_text.sort(key=len, reverse=True)
        primary_text = all_text[0]
        
        # Add unique parts from other extractions
        for text in all_text[1:]:
            # Simple deduplication - add unique lines
            primary_lines = set(primary_text.split('\n'))
            for line in text.split('\n'):
                if line.strip() and line not in primary_lines:
                    primary_text += '\n' + line
        
        return primary_text
    
    return ""

def main():
    """Main function to run enhanced debug processing"""
    
    # Look for test parcel map
    test_files = [
        "LOT 2 324 Dolan Rd Aerial Map.pdf",
        "test_parcel.pdf",
        "sample_parcel.pdf"
    ]
    
    test_file = None
    for filename in test_files:
        if os.path.exists(filename):
            test_file = filename
            break
    
    if not test_file:
        print("❌ No test parcel file found")
        print(f"   Looking for: {test_files}")
        return
    
    print(f"📄 Processing file: {test_file}")
    result = process_parcel_map_enhanced(test_file)
    
    if result and result.get('success'):
        print("\n✅ Enhanced debug processing completed successfully!")
    else:
        print("\n❌ Enhanced debug processing failed!")

if __name__ == "__main__":
    main() 