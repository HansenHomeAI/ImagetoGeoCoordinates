import os
import json
import uuid
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

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Initialize OCR readers lazily to avoid blocking server startup
easyocr_reader = None
geolocator = Nominatim(user_agent="ImagetoGeoCoordinates")

def get_easyocr_reader():
    global easyocr_reader
    if easyocr_reader is None:
        print("Initializing EasyOCR reader...")
        easyocr_reader = easyocr.Reader(['en'])
    return easyocr_reader

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf', 'heic'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class ParcelMapProcessor:
    def __init__(self):
        self.current_image = None
        self.extracted_text = ""
        self.detected_coordinates = []
        self.property_bounds = []
        
    def process_image(self, image_path):
        """Main processing pipeline for parcel maps"""
        try:
            # Step 1: Load and preprocess image
            self.current_image = self.load_image(image_path)
            if self.current_image is None:
                return {"error": "Could not load image"}
            
            # Step 2: Extract text using OCR
            self.extracted_text = self.extract_text_ocr(self.current_image)
            
            # Step 3: Find location clues in text
            location_info = self.extract_location_clues(self.extracted_text)
            
            # Step 4: Geocode location to get approximate coordinates
            base_coords = self.geocode_location(location_info)
            
            # Step 5: Detect shapes and property lines
            shapes = self.detect_property_boundaries(self.current_image)
            
            # Step 6: Convert detected shapes to coordinates
            coordinates = self.shapes_to_coordinates(shapes, base_coords)
            
            return {
                "success": True,
                "extracted_text": self.extracted_text,
                "location_info": location_info,
                "base_coordinates": base_coords,
                "property_coordinates": coordinates,
                "detected_shapes": len(shapes)
            }
            
        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}
    
    def load_image(self, image_path):
        """Load image from various formats"""
        try:
            file_ext = os.path.splitext(image_path)[1].lower()
            
            if file_ext == '.pdf':
                # Convert PDF to image
                pages = convert_from_path(image_path, first_page=1, last_page=1)
                if pages:
                    # Convert PIL to OpenCV format
                    pil_image = pages[0]
                    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    return opencv_image
            else:
                # Load regular image formats (including HEIC via pillow-heif)
                pil_image = Image.open(image_path)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                return opencv_image
                
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def extract_text_ocr(self, image):
        """Extract text using both Tesseract and EasyOCR"""
        try:
            # Use EasyOCR for better accuracy
            reader = get_easyocr_reader()
            results = reader.readtext(image)
            extracted_texts = [result[1] for result in results if result[2] > 0.5]  # confidence > 0.5
            
            # Also try Tesseract as backup
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                tesseract_text = pytesseract.image_to_string(pil_image)
                extracted_texts.append(tesseract_text)
            except:
                pass
            
            return " ".join(extracted_texts)
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def extract_location_clues(self, text):
        """Extract location information from OCR text"""
        location_info = {
            "streets": [],
            "coordinates": [],
            "county": None,
            "city": None,
            "state": None
        }
        
        try:
            # Find street names (common patterns)
            street_patterns = [
                r'([A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Place|Pl))',
                r'([A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln))'
            ]
            
            for pattern in street_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                location_info["streets"].extend(matches)
            
            # Find coordinate patterns
            coord_patterns = [
                r'(\d{1,3}°\d{1,2}\'[\d.]+\"[NS])',  # DMS format
                r'(\d{1,3}\.\d+°[NS])',  # Decimal degrees
                r'(\d{1,3}°\d{1,2}\'[\d.]+\"[EW])',  # DMS format
                r'(\d{1,3}\.\d+°[EW])',  # Decimal degrees
                r'([-]?\d{1,3}\.\d+,\s*[-]?\d{1,3}\.\d+)'  # Lat,Lon decimal
            ]
            
            for pattern in coord_patterns:
                matches = re.findall(pattern, text)
                location_info["coordinates"].extend(matches)
            
            # Find county, city, state
            county_match = re.search(r'([A-Z][a-z]+\s+County)', text, re.IGNORECASE)
            if county_match:
                location_info["county"] = county_match.group(1)
            
            # Remove duplicates
            location_info["streets"] = list(set(location_info["streets"]))
            location_info["coordinates"] = list(set(location_info["coordinates"]))
            
        except Exception as e:
            print(f"Location extraction error: {e}")
        
        return location_info
    
    def geocode_location(self, location_info):
        """Convert location clues to actual coordinates"""
        try:
            # Try to geocode using available information
            search_terms = []
            
            if location_info["county"]:
                search_terms.append(location_info["county"])
            
            if location_info["streets"]:
                search_terms.extend(location_info["streets"][:2])  # Use first 2 streets
            
            if search_terms:
                search_query = ", ".join(search_terms)
                location = geolocator.geocode(search_query, timeout=10)
                
                if location:
                    return {
                        "latitude": location.latitude,
                        "longitude": location.longitude,
                        "address": location.address
                    }
            
            # If no geocoding successful, return None
            return None
            
        except GeocoderTimedOut:
            print("Geocoding timeout")
            return None
        except Exception as e:
            print(f"Geocoding error: {e}")
            return None
    
    def detect_property_boundaries(self, image):
        """Detect property boundary lines and shapes"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and shape
            property_shapes = []
            min_area = 1000  # Minimum area threshold
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Filter for polygonal shapes (3+ vertices)
                    if len(approx) >= 3:
                        property_shapes.append(approx)
            
            return property_shapes
            
        except Exception as e:
            print(f"Shape detection error: {e}")
            return []
    
    def shapes_to_coordinates(self, shapes, base_coords):
        """Convert detected shapes to geographical coordinates"""
        coordinates_list = []
        
        try:
            if not base_coords or not shapes:
                return coordinates_list
            
            # For now, we'll create a simple mapping
            # In production, this would need more sophisticated coordinate transformation
            base_lat = base_coords["latitude"]
            base_lon = base_coords["longitude"]
            
            for shape in shapes:
                shape_coords = []
                
                # Convert pixel coordinates to rough lat/lon offsets
                for point in shape.reshape(-1, 2):
                    x, y = point
                    
                    # Very rough conversion (needs improvement for production)
                    # This assumes 1 pixel ≈ small lat/lon offset
                    lat_offset = (y - 500) * 0.0001  # Rough approximation
                    lon_offset = (x - 500) * 0.0001  # Rough approximation
                    
                    coord = {
                        "latitude": base_lat + lat_offset,
                        "longitude": base_lon + lon_offset,
                        "pixel_x": int(x),
                        "pixel_y": int(y)
                    }
                    shape_coords.append(coord)
                
                coordinates_list.append(shape_coords)
            
        except Exception as e:
            print(f"Coordinate conversion error: {e}")
        
        return coordinates_list

# Initialize processor
processor = ParcelMapProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            file.save(filepath)
            
            # Process the image
            result = processor.process_image(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 