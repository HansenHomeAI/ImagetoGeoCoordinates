# Technical Documentation - Image to Geo Coordinates System

## 🔧 Architecture Deep Dive

### System Design Principles

The Image to Geo Coordinates system follows a modular, pipeline-based architecture designed for accuracy, maintainability, and extensibility. Each component is loosely coupled and can be enhanced independently.

#### Core Design Patterns

1. **Pipeline Pattern**: Sequential processing stages with clear interfaces
2. **Strategy Pattern**: Multiple OCR engines with fallback mechanisms
3. **Observer Pattern**: Real-time progress tracking via WebSocket
4. **Factory Pattern**: Coordinate validation and correction system creation

### Component Interaction Diagram

```
[PDF Upload] 
    ↓
[Flask App] → [WebSocket] → [Client Updates]
    ↓
[OCR Processor] → [Text Extraction] → [Address Parser]
    ↓
[Shape Detector] → [Boundary Detection] → [Pixel Coordinates]
    ↓
[Coordinate Validator] → [Geocoding] → [Scale Analysis]
    ↓
[Coordinate Corrector] → [Transformation] → [Final Coordinates]
    ↓
[Accuracy Tester] → [Validation] → [Confidence Score]
```

## 📡 API Reference

### REST Endpoints

#### `POST /upload`
Uploads and processes a parcel map PDF file.

**Request:**
```http
POST /upload HTTP/1.1
Content-Type: multipart/form-data

file: [PDF file]
```

**Response:**
```json
{
  "status": "success",
  "message": "File processed successfully",
  "filename": "processed_map.pdf",
  "processing_time": 12.34,
  "data": {
    "coordinate_sets": [...],
    "accuracy_score": 0.667,
    "validation_results": {...}
  }
}
```

#### `GET /status`
Returns current processing status.

**Response:**
```json
{
  "status": "processing",
  "stage": "coordinate_validation",
  "progress": 75,
  "eta_seconds": 3
}
```

#### `GET /`
Serves the main web interface.

### WebSocket Events

#### Client → Server Events

**`connect`**
```javascript
socket.emit('connect');
```

#### Server → Client Events

**`processing_update`**
```javascript
socket.on('processing_update', (data) => {
  // data.stage, data.progress, data.message
});
```

**`processing_complete`**
```javascript
socket.on('processing_complete', (results) => {
  // Final processing results
});
```

**`processing_error`**
```javascript
socket.on('processing_error', (error) => {
  // Error information
});
```

## 🔍 Algorithm Details

### OCR Processing Pipeline

The OCR system uses a multi-engine approach for maximum text extraction accuracy.

### Shape Detection Algorithm

Property boundary detection uses advanced computer vision techniques with OpenCV.

### Geocoding Strategy Implementation

The geocoding system implements a sophisticated fallback strategy targeting Washington State addresses.

### Scale Analysis Algorithm

Scale bar detection and analysis from map text using regex patterns.

### Coordinate Transformation Mathematics

Precise coordinate transformation using WGS84 ellipsoid calculations.

## 🧪 Testing Framework

### Accuracy Testing Methodology

The accuracy testing framework implements multiple validation layers with weighted scoring:

- Geographic Reasonableness (25%)
- Distance Validation (25%)
- Property Size Validation (20%)
- Coordinate Clustering (15%)
- Reverse Geocoding (10%)
- Scale Consistency (5%)

## 🔒 Security Considerations

### File Upload Security

- File type validation (PDF only)
- File size limits (50MB max)
- Content scanning for malicious files
- Secure temporary file handling

### API Rate Limiting

- 100 requests per hour per IP
- 10 uploads per minute per IP
- WebSocket connection limits

## 📊 Monitoring & Logging

### Logging Configuration

- Rotating file handlers for application and error logs
- Structured logging with timestamps and context
- Debug mode for development environments

### Performance Metrics Collection

- Processing time tracking
- Accuracy score monitoring
- Error rate analysis
- Memory usage monitoring

## 🔧 Configuration Management

### Environment Configuration

Key configuration options:
- HOST, PORT, DEBUG mode
- File processing limits
- OCR and geocoding timeouts
- Logging levels

## 🚀 Deployment Strategies

### Docker Deployment

Complete containerization with system dependencies included.

### Production Configuration

- SSL/TLS configuration
- Database integration options
- Redis caching setup
- Load balancer configuration

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

Automated testing, security scanning, and deployment pipeline.

## 📈 Performance Optimization

### Memory Optimization

- Garbage collection monitoring
- Large image memory management
- Processing result caching

### Caching Strategy

- Redis-based result caching
- In-memory LRU caching for frequent operations
- Geocoding result caching

## 🔍 Debugging & Troubleshooting

### Debug Mode Configuration

Enhanced logging and intermediate result saving for debugging.

### Common Issues & Solutions

1. **Port Already in Use**: Kill existing processes
2. **OCR Timeout Issues**: Fallback strategy implementation
3. **Memory Leaks**: Explicit memory management
4. **Low Accuracy Results**: Input validation and preprocessing

---

*This technical documentation is maintained alongside the main README.md and should be updated whenever system components are modified.* 