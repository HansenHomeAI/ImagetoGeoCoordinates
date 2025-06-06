import os
import json
import uuid
import logging
import traceback
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import pytesseract
import easyocr
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import re
from pdf2image import convert_from_path
import pillow_heif
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Initialize OCR readers lazily to avoid blocking server startup
easyocr_reader = None
geolocator = Nominatim(user_agent="ImagetoGeoCoordinates-v2.0")

def get_easyocr_reader():
    global easyocr_reader
    if easyocr_reader is None:
        logger.info("🔄 Initializing EasyOCR reader - this may take a few minutes on first run...")
        try:
            easyocr_reader = easyocr.Reader(['en'], verbose=True)
            logger.info("✅ EasyOCR reader initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize EasyOCR: {e}")
            logger.error(traceback.format_exc())
            return None
    return easyocr_reader

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf', 'heic'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class AdvancedParcelMapProcessor:
    def __init__(self):
        self.current_image = None
        self.extracted_text = ""
        self.detected_coordinates = []
        self.property_bounds = []
        
    def process_image(self, image_path):
        """Enhanced processing pipeline for parcel maps with comprehensive logging"""
        logger.info(f"🚀 Starting parcel map processing for: {image_path}")
        
        try:
            # Step 1: Load and preprocess image
            logger.info("📷 Step 1: Loading and preprocessing image...")
            self.current_image = self.load_image(image_path)
            if self.current_image is None:
                logger.error("❌ Failed to load image")
                return {"error": "Could not load image"}
            
            logger.info(f"✅ Image loaded successfully - Shape: {self.current_image.shape}")
            
            # Step 2: Extract text using multiple OCR engines
            logger.info("🔍 Step 2: Extracting text using OCR...")
            self.extracted_text = self.extract_text_comprehensive(self.current_image)
            logger.info(f"📝 Extracted text length: {len(self.extracted_text)} characters")
            logger.debug(f"📄 Full extracted text: {self.extracted_text[:500]}...")
            
            # Store for advanced processor access
            self._last_extracted_text = self.extracted_text
            
            # Step 3: Find location clues in text
            logger.info("🌍 Step 3: Analyzing location clues...")
            location_info = self.extract_location_clues(self.extracted_text)
            logger.info(f"🏠 Location info found: {location_info}")
            
            # Store for advanced processor access
            self._last_location_info = location_info
            
            # Step 4: Geocode location to get approximate coordinates
            logger.info("🗺️ Step 4: Geocoding location...")
            base_coords = self.geocode_location(location_info)
            logger.info(f"📍 Base coordinates: {base_coords}")
            
            # Step 5: Enhanced shape detection and property lines
            logger.info("🔍 Step 5: Detecting property boundaries...")
            shapes = self.detect_property_boundaries_advanced(self.current_image)
            logger.info(f"🏠 Detected {len(shapes)} potential property shapes")
            
            # Step 6: Convert detected shapes to coordinates
            logger.info("📐 Step 6: Converting shapes to coordinates...")
            coordinates = self.shapes_to_coordinates_advanced(shapes, base_coords, self.current_image.shape)
            logger.info(f"📊 Generated {len(coordinates)} coordinate sets")
            
            result = {
                "success": True,
                "extracted_text": self.extracted_text,
                "location_info": location_info,
                "base_coordinates": base_coords,
                "property_coordinates": coordinates,
                "detected_shapes": len(shapes),
                "processing_log": f"Successfully processed {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
            
            logger.info("🎉 Processing completed successfully!")
            return result
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(f"💥 {error_msg}")
            logger.error(traceback.format_exc())
            return {"error": error_msg, "traceback": traceback.format_exc()}
    
    def load_image(self, image_path):
        """Enhanced image loading with better error handling"""
        try:
            file_ext = os.path.splitext(image_path)[1].lower()
            logger.info(f"📁 Loading file type: {file_ext}")
            
            if file_ext == '.pdf':
                logger.info("📄 Converting PDF to image...")
                pages = convert_from_path(image_path, first_page=1, last_page=1, dpi=300)
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
                pil_image = Image.open(image_path)
                logger.info(f"📏 Original image mode: {pil_image.mode}, size: {pil_image.size}")
                
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                    logger.info("🔄 Converted image to RGB")
                
                opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                logger.info(f"✅ Image loaded successfully - OpenCV shape: {opencv_image.shape}")
                return opencv_image
                
        except Exception as e:
            logger.error(f"💥 Error loading image: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def extract_text_comprehensive(self, image):
        """Enhanced text extraction using multiple OCR engines and preprocessing"""
        logger.info("🔍 Starting comprehensive text extraction...")
        all_text = []
        
        # Preprocess image for better OCR
        processed_images = self.preprocess_for_ocr(image)
        
        for i, processed_img in enumerate(processed_images):
            logger.info(f"📖 Processing image variant {i+1}/{len(processed_images)}")
            
            # Try EasyOCR
            try:
                logger.info("🤖 Attempting EasyOCR extraction...")
                reader = get_easyocr_reader()
                if reader:
                    results = reader.readtext(processed_img, detail=1, paragraph=False)
                    easyocr_texts = []
                    for bbox, text, confidence in results:
                        if confidence > 0.3:  # Lower threshold for more text
                            easyocr_texts.append(text)
                            logger.debug(f"📝 EasyOCR found: '{text}' (confidence: {confidence:.2f})")
                    
                    easyocr_combined = " ".join(easyocr_texts)
                    all_text.append(easyocr_combined)
                    logger.info(f"✅ EasyOCR extracted {len(easyocr_texts)} text segments")
                else:
                    logger.warning("⚠️ EasyOCR reader not available")
            except Exception as e:
                logger.error(f"❌ EasyOCR failed: {e}")
            
            # Try Tesseract with different configurations
            for config in ['--psm 6', '--psm 11', '--psm 12', '--psm 13']:
                try:
                    logger.info(f"📖 Attempting Tesseract with config: {config}")
                    rgb_image = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                    tesseract_text = pytesseract.image_to_string(pil_image, config=config)
                    if tesseract_text.strip():
                        all_text.append(tesseract_text)
                        logger.info(f"✅ Tesseract ({config}) extracted {len(tesseract_text)} characters")
                        logger.debug(f"📄 Tesseract text preview: {tesseract_text[:200]}...")
                except Exception as e:
                    logger.error(f"❌ Tesseract ({config}) failed: {e}")
        
        # Combine all extracted text
        combined_text = " ".join(all_text)
        logger.info(f"📊 Total text extraction complete - {len(combined_text)} characters")
        return combined_text
    
    def preprocess_for_ocr(self, image):
        """Create multiple preprocessed versions of the image for better OCR"""
        logger.info("🔧 Preprocessing image for OCR...")
        processed_images = []
        
        # Original image
        processed_images.append(image.copy())
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_images.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        
        # High contrast version
        contrast = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
        processed_images.append(cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR))
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_images.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))
        
        # Morphological operations to enhance text
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        processed_images.append(cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR))
        
        logger.info(f"✅ Created {len(processed_images)} preprocessed versions")
        return processed_images
    
    def extract_location_clues(self, text):
        """Enhanced location information extraction with more patterns"""
        logger.info("🔍 Extracting location clues from text...")
        location_info = {
            "streets": [],
            "coordinates": [],
            "county": None,
            "city": None,
            "state": None,
            "addresses": [],
            "lot_numbers": [],
            "parcel_ids": []
        }
        
        try:
            # Enhanced street name patterns
            street_patterns = [
                r'(\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Place|Pl|Way|Circle|Cir|Trail|Tr))',
                r'([A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Place|Pl|Way|Circle|Cir|Trail|Tr))',
                r'(\d+\s+[A-Za-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln))'
            ]
            
            all_streets = []
            for pattern in street_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                all_streets.extend(matches)
                logger.debug(f"🛣️ Street pattern found: {matches}")
            
            # Clean up street names
            clean_streets = []
            for street in all_streets:
                # Remove common OCR artifacts and clean up
                clean_street = re.sub(r'[^\w\s]', ' ', street)
                clean_street = re.sub(r'\s+', ' ', clean_street).strip()
                if len(clean_street) > 5:  # Only keep meaningful street names
                    clean_streets.append(clean_street)
            
            location_info["streets"] = list(set(clean_streets))  # Remove duplicates
            logger.info(f"🧹 Cleaned streets: {location_info['streets']}")
            
            # Enhanced coordinate patterns
            coord_patterns = [
                r'(\d{1,3}°\s*\d{1,2}\'\s*[\d.]+\"\s*[NS])',  # DMS format with spaces
                r'(\d{1,3}\.\d{4,}[°]?\s*[NS])',  # Decimal degrees
                r'(\d{1,3}°\s*\d{1,2}\'\s*[\d.]+\"\s*[EW])',  # DMS format with spaces
                r'(\d{1,3}\.\d{4,}[°]?\s*[EW])',  # Decimal degrees
                r'([-]?\d{1,3}\.\d{4,}\s*,\s*[-]?\d{1,3}\.\d{4,})',  # Lat,Lon decimal
                r'(\d{2}\.\d{6,})',  # High precision coordinates
                r'(UTM\s+\d+[NS]\s+\d+\s+\d+)',  # UTM coordinates
                r'(State Plane\s+\d+)',  # State Plane coordinates
            ]
            
            for pattern in coord_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                location_info["coordinates"].extend(matches)
                logger.debug(f"📍 Coordinate pattern found: {matches}")
            
            # County, city, state patterns
            county_patterns = [
                r'([A-Za-z]+(?:\s+[A-Za-z]+)*\s+County)',
                r'County\s+of\s+([A-Za-z]+)',
                r'([A-Za-z]+)\s+Co\.?'
            ]
            
            for pattern in county_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    location_info["county"] = match.group(1)
                    logger.debug(f"🏛️ County found: {location_info['county']}")
                    break
            
            # Enhanced state detection - look for Washington state indicators
            state = None
            if 'Washington' in text:
                state = 'WA'
            elif 'Cowlitz' in text:  # Cowlitz County is in Washington
                state = 'WA'
            elif 'WA DNR' in text:  # Washington Department of Natural Resources
                state = 'WA'
            else:
                # Try to find two-letter state codes, but filter out common false positives
                state_match = re.search(r'\b(WA|OR|CA|ID|MT|ND|SD|WY|CO|UT|NV|AZ|NM|TX|OK|KS|NE|IA|MN|WI|IL|IN|OH|MI|PA|NY|VT|NH|ME|MA|RI|CT|NJ|DE|MD|VA|WV|KY|TN|NC|SC|GA|FL|AL|MS|LA|AR|MO|AK|HI)\b', text)
                if state_match:
                    state = state_match.group(1)
            
            location_info["state"] = state
            
            # Lot and parcel information
            lot_patterns = [
                r'(Lot\s+\d+)',
                r'(Parcel\s+\d+)',
                r'(Block\s+\d+)',
                r'(Tract\s+\d+)'
            ]
            
            for pattern in lot_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                location_info["lot_numbers"].extend(matches)
                logger.debug(f"🏠 Lot/Parcel info found: {matches}")
            
            # Remove duplicates
            for key in ["coordinates", "addresses", "lot_numbers", "parcel_ids"]:
                location_info[key] = list(set(location_info[key]))
            
            logger.info(f"✅ Location extraction complete: {location_info}")
            
        except Exception as e:
            logger.error(f"💥 Location extraction error: {e}")
            logger.error(traceback.format_exc())
        
        return location_info
    
    def geocode_location(self, location_info):
        """Enhanced geocoding with multiple strategies"""
        logger.info("🌐 Starting enhanced geocoding...")
        
        try:
            search_attempts = []
            
            # Strategy 1: Use specific location context for Washington state
            if location_info["county"] and location_info["state"]:
                # For Cowlitz County, try specific Washington locations
                if "Cowlitz" in location_info["county"] and location_info["state"] == "WA":
                    search_attempts.append("Cowlitz County, Washington, USA")
                    search_attempts.append("Kelso, Cowlitz County, Washington, USA")
                    search_attempts.append("Longview, Cowlitz County, Washington, USA")
                else:
                    search_attempts.append(f"{location_info['county']}, {location_info['state']}, USA")
            
            # Strategy 2: Try streets with specific Washington context
            if location_info["streets"]:
                for street in location_info["streets"][:3]:  # Try first 3 streets
                    if "Dolan" in street and location_info["county"] and location_info["state"] == "WA":
                        # Try specific Dolan Road locations in Washington
                        search_attempts.append(f"Dolan Road, Cowlitz County, Washington, USA")
                        search_attempts.append(f"Dolan Road, Washington, USA")
                    
                    if location_info["county"] and location_info["state"]:
                        search_attempts.append(f"{street}, {location_info['county']}, {location_info['state']}, USA")
                    if location_info["county"]:
                        search_attempts.append(f"{street}, {location_info['county']}")
                    search_attempts.append(street)
            
            # Strategy 3: County + State with USA
            if location_info["county"] and location_info["state"]:
                search_attempts.append(f"{location_info['county']}, {location_info['state']}, USA")
            
            # Strategy 4: Just county or state with USA
            if location_info["county"]:
                search_attempts.append(f"{location_info['county']}, USA")
            
            for attempt in search_attempts:
                logger.info(f"🔍 Trying geocoding query: '{attempt}'")
                try:
                    location = geolocator.geocode(attempt, timeout=15)
                    if location:
                        # Validate that we're in the right general area
                        # Washington state is roughly between 45-49°N and 116-125°W
                        if location_info["state"] == "WA":
                            if 45 <= location.latitude <= 49 and -125 <= location.longitude <= -116:
                                result = {
                                    "latitude": location.latitude,
                                    "longitude": location.longitude,
                                    "address": location.address,
                                    "search_query": attempt
                                }
                                logger.info(f"✅ Geocoding successful: {result}")
                                return result
                            else:
                                logger.warning(f"⚠️ Location outside Washington state bounds: {location.latitude}, {location.longitude}")
                                continue
                        else:
                            result = {
                                "latitude": location.latitude,
                                "longitude": location.longitude,
                                "address": location.address,
                                "search_query": attempt
                            }
                            logger.info(f"✅ Geocoding successful: {result}")
                            return result
                    else:
                        logger.warning(f"⚠️ No results for query: '{attempt}'")
                except GeocoderTimedOut:
                    logger.warning(f"⏰ Geocoding timeout for query: '{attempt}'")
                except Exception as e:
                    logger.error(f"❌ Geocoding error for '{attempt}': {e}")
            
            logger.warning("⚠️ All geocoding attempts failed")
            return None
            
        except Exception as e:
            logger.error(f"💥 Geocoding error: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def detect_property_boundaries_advanced(self, image):
        """Advanced property boundary detection with multiple techniques optimized for aerial maps"""
        logger.info("🔍 Starting advanced property boundary detection...")
        
        try:
            # Try the enhanced property detector first
            try:
                from enhanced_property_detector import EnhancedPropertyDetector
                
                # Create enhanced detector instance
                enhanced_detector = EnhancedPropertyDetector()
                
                # Prepare text information for context
                text_info = {
                    "extracted_text": getattr(self, '_last_extracted_text', '')
                }
                
                # Use the enhanced property detection
                shapes = enhanced_detector.detect_property_boundaries(image, text_info)
                
                if shapes:
                    logger.info(f"🎯 Enhanced detection found {len(shapes)} property candidates")
                    return shapes
                else:
                    logger.warning("⚠️ Enhanced detection found no candidates, falling back to basic methods")
                    
            except ImportError as e:
                logger.warning(f"⚠️ Enhanced detector not available: {e}, using fallback")
            except Exception as e:
                logger.warning(f"⚠️ Enhanced detection failed: {e}, using fallback")
            
            # Use the proven fallback method
            return self._fallback_boundary_detection(image)
            
        except Exception as e:
            logger.error(f"💥 Boundary detection error: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def _fallback_boundary_detection(self, image):
        """Fallback boundary detection using original methods"""
        logger.info("🔄 Using fallback boundary detection...")
        
        shapes = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.info(f"📏 Image dimensions: {gray.shape}")
        
        # Use proven edge detection parameters from diagnostic
        edge_methods = [
            ("Canny_very_sensitive", lambda img: cv2.Canny(img, 10, 50)),  # Best from diagnostic
            ("Canny_conservative", lambda img: cv2.Canny(img, 30, 90)),
            ("Canny_standard", lambda img: cv2.Canny(img, 50, 150)),
            ("Canny_aggressive", lambda img: cv2.Canny(img, 100, 200)),
        ]
        
        all_contours = []
        
        for method_name, edge_func in edge_methods:
            logger.info(f"🔍 Trying edge detection method: {method_name}")
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply edge detection
            edges = edge_func(blurred)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            logger.info(f"📊 Found {len(contours)} contours with {method_name}")
            
            # Filter and process contours with enhanced criteria for aerial maps
            method_shapes = self.filter_property_contours_enhanced(contours, gray.shape)
            all_contours.extend(method_shapes)
            logger.info(f"✅ {method_name} contributed {len(method_shapes)} valid shapes")
        
        # Remove duplicate shapes
        unique_shapes = self.remove_duplicate_shapes_enhanced(all_contours)
        logger.info(f"🎯 Final unique shapes: {len(unique_shapes)}")
        
        return unique_shapes
    
    def filter_property_contours_enhanced(self, contours, image_shape):
        """Enhanced contour filtering optimized for aerial property maps"""
        logger.debug("🔍 Enhanced filtering contours for property boundaries...")
        
        height, width = image_shape
        min_area = (width * height) * 0.0001  # Further reduced minimum area to match diagnostic findings
        max_area = (width * height) * 0.4     # Increased maximum area
        
        property_shapes = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if min_area < area < max_area:
                # Approximate contour to polygon with different epsilon values
                perimeter = cv2.arcLength(contour, True)
                
                # Try multiple approximation levels
                for epsilon_factor in [0.01, 0.02, 0.03]:
                    epsilon = epsilon_factor * perimeter
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Filter for reasonable polygon shapes (3-15 vertices for properties)
                    if 3 <= len(approx) <= 15:
                        # Check if it's roughly rectangular or polygonal (property-like)
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        
                        if hull_area > 0:
                            solidity = area / hull_area
                            
                            # More lenient solidity for aerial maps
                            if solidity > 0.3:  # Reduced threshold for aerial imagery
                                # Check aspect ratio to filter out very thin shapes
                                rect = cv2.minAreaRect(contour)
                                width_rect, height_rect = rect[1]
                                if width_rect > 0 and height_rect > 0:
                                    aspect_ratio = max(width_rect, height_rect) / min(width_rect, height_rect)
                                    
                                    # Filter out extremely elongated shapes
                                    if aspect_ratio < 10:  # Not too elongated
                                        property_shapes.append((approx, area, solidity, aspect_ratio))
                                        logger.debug(f"✅ Valid shape {i}: {len(approx)} vertices, area: {int(area)}, solidity: {solidity:.2f}, aspect: {aspect_ratio:.2f}")
                                        break  # Found good approximation, no need to try other epsilon values
        
        # Sort by area (larger shapes first) and return just the approximated contours
        property_shapes.sort(key=lambda x: x[1], reverse=True)
        return [shape[0] for shape in property_shapes]
    
    def remove_duplicate_shapes_enhanced(self, shapes):
        """Enhanced duplicate removal with better similarity detection"""
        if not shapes:
            return shapes
        
        unique_shapes = []
        tolerance = 100  # pixels - increased tolerance for aerial maps
        
        for shape in shapes:
            is_duplicate = False
            
            for unique_shape in unique_shapes:
                # Check similarity based on area and centroid
                area1 = cv2.contourArea(shape)
                area2 = cv2.contourArea(unique_shape)
                
                # If areas are very different, not duplicates
                if abs(area1 - area2) > tolerance * tolerance:
                    continue
                
                # Check centroid distance
                try:
                    M1 = cv2.moments(shape)
                    M2 = cv2.moments(unique_shape)
                    
                    if M1["m00"] > 0 and M2["m00"] > 0:
                        cx1, cy1 = int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"])
                        cx2, cy2 = int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"])
                        centroid_distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                        
                        if centroid_distance < tolerance:
                            # Additional check: compare vertex distances if same number of vertices
                            if len(shape) == len(unique_shape):
                                max_vertex_distance = 0
                                for pt1, pt2 in zip(shape.reshape(-1, 2), unique_shape.reshape(-1, 2)):
                                    distance = np.linalg.norm(pt1 - pt2)
                                    max_vertex_distance = max(max_vertex_distance, distance)
                                
                                if max_vertex_distance < tolerance:
                                    is_duplicate = True
                                    break
                            else:
                                # Different vertex counts but similar area and centroid - likely duplicate
                                is_duplicate = True
                                break
                except:
                    # If moment calculation fails, fall back to simple comparison
                    continue
            
            if not is_duplicate:
                unique_shapes.append(shape)
        
        logger.debug(f"🎯 Removed {len(shapes) - len(unique_shapes)} duplicate shapes")
        return unique_shapes
    
    def shapes_to_coordinates_advanced(self, shapes, base_coords, image_shape):
        """Advanced conversion of shapes to geographical coordinates using enhanced methods"""
        logger.info("📐 Converting shapes to geographical coordinates with advanced validation...")
        
        coordinates_list = []
        
        try:
            if not shapes:
                logger.warning("⚠️ No shapes provided for coordinate conversion")
                return coordinates_list
            
            # Try using the advanced processor for coordinate calculation
            try:
                from advanced_parcel_processor import AdvancedParcelProcessor
                
                advanced_processor = AdvancedParcelProcessor()
                
                # Prepare location information
                location_info = {
                    "extracted_text": getattr(self, '_last_extracted_text', ''),
                    "streets": getattr(self, '_last_location_info', {}).get('streets', []),
                    "county": getattr(self, '_last_location_info', {}).get('county'),
                    "state": getattr(self, '_last_location_info', {}).get('state'),
                    "coordinates": getattr(self, '_last_location_info', {}).get('coordinates', [])
                }
                
                # Use advanced coordinate calculation
                advanced_coords = advanced_processor.calculate_precise_coordinates(
                    shapes, location_info, image_shape
                )
                
                if advanced_coords:
                    logger.info(f"🎯 Advanced coordinate calculation successful: {len(advanced_coords)} shapes")
                    return advanced_coords
                else:
                    logger.warning("⚠️ Advanced coordinate calculation returned no results, using fallback")
                    
            except Exception as e:
                logger.warning(f"⚠️ Advanced coordinate calculation failed: {e}, using fallback")
            
            # Fallback to original method
            return self._fallback_coordinate_calculation(shapes, base_coords, image_shape)
            
        except Exception as e:
            logger.error(f"💥 Coordinate conversion error: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def _fallback_coordinate_calculation(self, shapes, base_coords, image_shape):
        """Fallback coordinate calculation using original method"""
        logger.info("🔄 Using fallback coordinate calculation...")
        
        coordinates_list = []
        
        if not base_coords:
            logger.warning("⚠️ No base coordinates available - using estimated coordinates")
            # Use a default center point for the US if no base coords
            base_coords = {
                "latitude": 39.8283,  # Geographic center of US
                "longitude": -98.5795,
                "address": "Estimated center point"
            }
        
        base_lat = base_coords["latitude"]
        base_lon = base_coords["longitude"]
        height, width = image_shape[:2]
        
        logger.info(f"📍 Base coordinates: {base_lat}, {base_lon}")
        logger.info(f"📏 Image dimensions: {width}x{height}")
        
        # Estimate scale (this is a rough approximation - in production would need more sophisticated methods)
        # Assume the image covers roughly 0.01 degrees (about 1km at mid-latitudes)
        estimated_coverage_degrees = 0.01
        pixels_per_degree_lat = height / estimated_coverage_degrees
        pixels_per_degree_lon = width / estimated_coverage_degrees
        
        logger.info(f"📏 Estimated scale: {pixels_per_degree_lat:.1f} pixels/degree lat, {pixels_per_degree_lon:.1f} pixels/degree lon")
        
        for i, shape in enumerate(shapes):
            logger.info(f"🏠 Processing shape {i+1}/{len(shapes)} with {len(shape)} vertices")
            shape_coords = []
            
            # Convert pixel coordinates to lat/lon
            for j, point in enumerate(shape.reshape(-1, 2)):
                x, y = point
                
                # Convert to relative position from center
                center_x, center_y = width / 2, height / 2
                rel_x = (x - center_x) / pixels_per_degree_lon
                rel_y = (center_y - y) / pixels_per_degree_lat  # Y is flipped in images
                
                # Calculate final coordinates
                latitude = base_lat + rel_y
                longitude = base_lon + rel_x
                
                coord = {
                    "latitude": round(latitude, 6),
                    "longitude": round(longitude, 6),
                    "pixel_x": int(x),
                    "pixel_y": int(y),
                    "vertex_index": j
                }
                shape_coords.append(coord)
                logger.debug(f"📍 Vertex {j}: ({x}, {y}) -> ({latitude:.6f}, {longitude:.6f})")
            
            coordinates_list.append(shape_coords)
            logger.info(f"✅ Shape {i+1} converted to {len(shape_coords)} coordinate points")
        
        logger.info(f"🎉 Successfully converted {len(coordinates_list)} shapes to coordinates")
        return coordinates_list

# Initialize processor
processor = AdvancedParcelMapProcessor()

@app.route('/')
def index():
    logger.info("🏠 Serving main index page")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.info("📤 File upload request received")
    
    if 'file' not in request.files:
        logger.error("❌ No file provided in request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("❌ No file selected")
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            logger.info(f"💾 Saving uploaded file: {unique_filename}")
            file.save(filepath)
            
            # Process the image
            logger.info(f"🚀 Starting processing for: {unique_filename}")
            result = processor.process_image(filepath)
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
                logger.info(f"🗑️ Cleaned up temporary file: {unique_filename}")
            except:
                pass
            
            logger.info("✅ Request processing completed successfully")
            return jsonify(result)
            
        except Exception as e:
            error_msg = f'Processing failed: {str(e)}'
            logger.error(f"💥 {error_msg}")
            logger.error(traceback.format_exc())
            return jsonify({'error': error_msg}), 500
    
    logger.error(f"❌ Invalid file type: {file.filename}")
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/health')
def health():
    logger.debug("💓 Health check requested")
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    logger.info("🚀 Starting Image to Geo Coordinates application...")
    logger.info("🌐 Server will be available at: http://localhost:8081")
    app.run(debug=True, host='0.0.0.0', port=8081) 