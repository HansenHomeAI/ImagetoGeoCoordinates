# ImagetoGeoCoordinates

A sophisticated Flask web application for extracting geographical coordinates from parcel plat maps and property documents. This tool uses advanced computer vision, OCR, and geocoding to automatically analyze uploaded images and convert property boundaries into precise coordinate data.

## 🚀 Features

- **Multi-format Support**: PDF, JPEG, PNG, HEIC, TIFF, and more
- **Advanced OCR**: Dual OCR engines (EasyOCR + Tesseract) for maximum text extraction accuracy
- **Computer Vision**: Automated property boundary detection using OpenCV
- **Geocoding**: Intelligent location identification from street names, counties, and coordinates
- **Modern Web Interface**: Responsive drag-and-drop interface with real-time processing
- **Coordinate Extraction**: Converts detected shapes to precise latitude/longitude coordinates
- **Visual Feedback**: Live processing status and detailed results display

## 🛠️ Installation

### Prerequisites

**System Dependencies:**
```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# Windows
# Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
# Download and install Poppler from: https://poppler.freedesktop.org/
```

**Python Setup:**
```bash
git clone https://github.com/HansenHomeAI/ImagetoGeoCoordinates.git
cd ImagetoGeoCoordinates

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

1. **Start the application:**
   ```bash
   source venv/bin/activate
   python app.py
   ```

2. **Open your browser:**
   Navigate to `http://localhost:8080`

3. **Upload a parcel map:**
   - Drag and drop your image/PDF into the upload area
   - Wait for processing (first run may take longer due to model downloads)
   - View extracted coordinates and location information

## 📋 Usage

### Supported File Types
- **Images**: JPEG, PNG, HEIC, TIFF, BMP, GIF
- **Documents**: PDF (first page)

### Processing Pipeline
1. **Image Loading**: Converts various formats to OpenCV-compatible format
2. **Text Extraction**: Uses EasyOCR and Tesseract to extract all text
3. **Location Analysis**: Identifies street names, counties, and coordinate references
4. **Geocoding**: Converts location clues to approximate coordinates
5. **Boundary Detection**: Uses computer vision to find property lines
6. **Coordinate Mapping**: Transforms pixel coordinates to geographical coordinates

### API Endpoints
- `GET /` - Main web interface
- `POST /upload` - File upload and processing
- `GET /health` - Health check endpoint

## 🔧 Development

### Project Structure
```
ImagetoGeoCoordinates/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── setup.py           # Setup and testing script
├── templates/
│   └── index.html     # Web interface
├── uploads/           # Temporary file storage
├── processed/         # Processed file storage
└── static/           # Static assets
```

### Key Components

**ParcelMapProcessor Class:**
- `load_image()` - Multi-format image loading
- `extract_text_ocr()` - Dual OCR text extraction
- `extract_location_clues()` - Pattern matching for location data
- `geocode_location()` - Address to coordinate conversion
- `detect_property_boundaries()` - Computer vision shape detection
- `shapes_to_coordinates()` - Coordinate transformation

### Testing
```bash
# Run setup script to verify installation
python setup.py

# Test with simplified app
python test_app.py
```

## 🎯 Production Deployment

This application is designed for local development and testing. For production deployment:

1. **AWS Lambda**: The codebase is structured for easy Lambda deployment
2. **Docker**: Consider containerization for consistent environments
3. **Security**: Add authentication and file validation
4. **Scaling**: Implement queue-based processing for large files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🔮 Future Enhancements

- [ ] Machine learning model for improved boundary detection
- [ ] Support for multi-page PDF processing
- [ ] Integration with GIS databases
- [ ] Batch processing capabilities
- [ ] Export to various coordinate formats (KML, GeoJSON, etc.)
- [ ] Advanced coordinate transformation algorithms
- [ ] Real-time collaboration features

## 🐛 Troubleshooting

**Common Issues:**

1. **Port 5000 in use**: The app uses port 8080 to avoid macOS AirPlay conflicts
2. **EasyOCR slow startup**: First run downloads models (~100MB), subsequent runs are faster
3. **Memory usage**: Large images may require significant RAM for processing
4. **Coordinate accuracy**: Current implementation provides rough approximations; production would need surveyor-grade algorithms

**Getting Help:**
- Check the [Issues](https://github.com/HansenHomeAI/ImagetoGeoCoordinates/issues) page
- Review the setup script output for dependency issues
- Ensure all system dependencies are properly installed 