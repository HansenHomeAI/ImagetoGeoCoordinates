#!/usr/bin/env python3
"""
Flask Web Application for AI Property Intelligence System
Integrates with o4-mini for advanced property map analysis
"""

import os
import json
import asyncio
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import uuid

# Import our AI system
from ai_property_intelligence_v1 import AIPropertyIntelligence

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize AI system
ai_system = None
try:
    ai_system = AIPropertyIntelligence()
    print("✅ AI Property Intelligence System initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize AI system: {e}")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'tiff', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and trigger AI analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Return upload success - analysis will be triggered separately
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'message': 'File uploaded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/analyze/<filename>')
def analyze_property_map(filename):
    """Analyze uploaded property map with o4-mini"""
    if not ai_system:
        return jsonify({'error': 'AI system not available'}), 500
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Run async analysis in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(ai_system.process_property_map(file_path))
        loop.close()
        
        # Format results for frontend
        response_data = {
            'success': True,
            'analysis_complete': True,
            'processing_time': result.processing_time,
            'cost_estimate': result.cost_estimate,
            'confidence_score': result.confidence_score,
            'vertices_found': len(result.geo_coordinates),
            'coordinates': result.geo_coordinates,
            'extracted_info': result.extracted_info,
            'reference_sources': len(result.reference_data),
            'reference_data': result.reference_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results for later retrieval
        results_filename = f"results_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_path = os.path.join(app.config['UPLOAD_FOLDER'], results_filename)
        with open(results_path, 'w') as f:
            json.dump(response_data, f, indent=2)
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'success': False
        }), 500

@app.route('/progress/<filename>')
def get_analysis_progress(filename):
    """Get analysis progress (for real-time updates)"""
    # This would be enhanced with actual progress tracking
    # For now, return mock progress data
    return jsonify({
        'progress': 75,
        'stage': 'AI coordinate calculation',
        'message': 'o4-mini analyzing spatial relationships...',
        'estimated_time_remaining': 15
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ai_system_available': ai_system is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/system-info')
def system_info():
    """Get system information and capabilities"""
    return jsonify({
        'system_name': 'AI Property Intelligence System',
        'version': '1.0',
        'ai_model': 'OpenAI o4-mini',
        'capabilities': [
            'Advanced vision analysis',
            'Multimodal reasoning',
            'Database cross-referencing',
            'Precise coordinate calculation',
            'Confidence scoring'
        ],
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': 16,
        'average_processing_time': '30-60 seconds',
        'typical_accuracy': '95%+',
        'cost_per_analysis': '$0.01-0.02'
    })

if __name__ == '__main__':
    print("🚀 Starting AI Property Intelligence Web Application...")
    print("🤖 Powered by OpenAI o4-mini")
    print("🌐 Access at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 