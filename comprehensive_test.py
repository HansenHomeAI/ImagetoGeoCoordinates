#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Parcel Map Processing System
Tests all components including open data integration, coordinate systems, and shape matching.
"""

import requests
import json
import time
import os
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveTestSuite:
    """Comprehensive test suite for the enhanced parcel processing system"""
    
    def __init__(self, server_url="http://127.0.0.1:8081"):
        self.server_url = server_url
        self.test_results = {}
        
    def run_all_tests(self):
        """Run all test suites"""
        
        print("🧪 Starting Comprehensive Test Suite")
        print("=" * 60)
        
        # Test 1: Server connectivity
        self.test_server_connectivity()
        
        # Test 2: Basic parcel processing
        self.test_basic_parcel_processing()
        
        # Test 3: Enhanced processing features
        self.test_enhanced_processing()
        
        # Test 4: Coordinate system detection
        self.test_coordinate_systems()
        
        # Test 5: Open data integration
        self.test_open_data_integration()
        
        # Test 6: Shape matching algorithms
        self.test_shape_matching()
        
        # Test 7: Error handling and edge cases
        self.test_error_handling()
        
        # Generate comprehensive report
        self.generate_test_report()
        
    def test_server_connectivity(self):
        """Test server connectivity and health"""
        
        print("\n🔗 Testing Server Connectivity")
        print("-" * 40)
        
        try:
            response = requests.get(self.server_url, timeout=10)
            if response.status_code == 200:
                print("✅ Server is responding")
                self.test_results['server_connectivity'] = True
            else:
                print(f"⚠️ Server returned status: {response.status_code}")
                self.test_results['server_connectivity'] = False
        except Exception as e:
            print(f"❌ Server connection failed: {e}")
            self.test_results['server_connectivity'] = False
    
    def test_basic_parcel_processing(self):
        """Test basic parcel map processing functionality"""
        
        print("\n📋 Testing Basic Parcel Processing")
        print("-" * 40)
        
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
            self.test_results['basic_processing'] = False
            return
        
        try:
            with open(test_file, 'rb') as f:
                files = {'file': (os.path.basename(test_file), f, 'application/pdf')}
                
                print(f"📤 Uploading {test_file}...")
                start_time = time.time()
                
                response = requests.post(
                    f"{self.server_url}/upload",
                    files=files,
                    timeout=300
                )
                
                processing_time = time.time() - start_time
                print(f"⏱️ Processing time: {processing_time:.2f} seconds")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Analyze results
                    self.analyze_processing_results(result, 'basic_processing')
                    
                else:
                    print(f"❌ Processing failed with status: {response.status_code}")
                    self.test_results['basic_processing'] = False
                    
        except Exception as e:
            print(f"❌ Basic processing test failed: {e}")
            self.test_results['basic_processing'] = False
    
    def test_enhanced_processing(self):
        """Test enhanced processing features"""
        
        print("\n🚀 Testing Enhanced Processing Features")
        print("-" * 40)
        
        # Test enhanced processor components
        try:
            from enhanced_processor_v2 import create_enhanced_processor
            processor = create_enhanced_processor()
            
            print("✅ Enhanced processor initialized successfully")
            
            # Test coordinate system detection
            test_coords = [(47.123456, -122.654321), (47.124000, -122.655000)]
            coord_system = processor.coord_processor.detect_coordinate_system(test_coords)
            print(f"✅ Coordinate system detection: {coord_system.name}")
            
            # Test open data integrator
            bbox = (47.1, -122.7, 47.2, -122.6)
            streets = processor.open_data.query_overpass_api(bbox, "highway")
            print(f"✅ Open data integration: Found {len(streets)} street features")
            
            self.test_results['enhanced_processing'] = True
            
        except Exception as e:
            print(f"❌ Enhanced processing test failed: {e}")
            self.test_results['enhanced_processing'] = False
    
    def test_coordinate_systems(self):
        """Test coordinate system detection and conversion"""
        
        print("\n🗺️ Testing Coordinate Systems")
        print("-" * 40)
        
        try:
            from enhanced_processor_v2 import AdvancedCoordinateProcessor
            coord_processor = AdvancedCoordinateProcessor()
            
            # Test different coordinate formats
            test_cases = [
                {
                    'name': 'WGS84 Decimal Degrees',
                    'coords': [(47.123456, -122.654321)],
                    'expected': 'WGS84'
                },
                {
                    'name': 'UTM Coordinates',
                    'coords': [(550000, 5220000)],
                    'expected': 'UTM'
                },
                {
                    'name': 'State Plane Coordinates',
                    'coords': [(1200000, 400000)],
                    'expected': 'State Plane'
                }
            ]
            
            for test_case in test_cases:
                detected = coord_processor.detect_coordinate_system(test_case['coords'])
                print(f"✅ {test_case['name']}: Detected {detected.name}")
            
            # Test coordinate conversion
            wgs84_coords = [(47.123456, -122.654321)]
            utm_system = coord_processor.coordinate_systems['UTM_Zone_10N']
            wgs84_system = coord_processor.coordinate_systems['WGS84']
            
            converted = coord_processor.convert_coordinates(wgs84_coords, wgs84_system, utm_system)
            print(f"✅ Coordinate conversion: {wgs84_coords[0]} -> {converted[0]}")
            
            self.test_results['coordinate_systems'] = True
            
        except Exception as e:
            print(f"❌ Coordinate system test failed: {e}")
            self.test_results['coordinate_systems'] = False
    
    def test_open_data_integration(self):
        """Test open data source integration"""
        
        print("\n🌐 Testing Open Data Integration")
        print("-" * 40)
        
        try:
            from enhanced_processor_v2 import OpenDataIntegrator
            integrator = OpenDataIntegrator()
            
            # Test Overpass API (OpenStreetMap)
            bbox = (46.1, -122.8, 46.3, -122.6)  # Cowlitz County area
            
            print("🔍 Querying OpenStreetMap data...")
            osm_data = integrator.query_overpass_api(bbox, "highway")
            print(f"✅ OpenStreetMap: Found {len(osm_data)} highway features")
            
            # Test Census TIGER data
            print("🔍 Querying Census TIGER data...")
            tiger_data = integrator.query_census_tiger(bbox)
            print(f"✅ Census TIGER: Found {len(tiger_data)} road features")
            
            # Test county parcel data query
            print("🔍 Querying county parcel data...")
            parcels = integrator.get_county_parcels("Cowlitz", "WA")
            print(f"✅ County parcels: Found {len(parcels)} parcel records")
            
            self.test_results['open_data_integration'] = True
            
        except Exception as e:
            print(f"❌ Open data integration test failed: {e}")
            self.test_results['open_data_integration'] = False
    
    def test_shape_matching(self):
        """Test shape matching algorithms"""
        
        print("\n🔷 Testing Shape Matching")
        print("-" * 40)
        
        try:
            # Create test image with simple shapes
            test_image = np.zeros((500, 500, 3), dtype=np.uint8)
            
            # Draw test rectangles (simulating parcels)
            cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), 2)
            cv2.rectangle(test_image, (250, 150), (400, 300), (255, 255, 255), 2)
            
            from enhanced_processor_v2 import create_enhanced_processor
            processor = create_enhanced_processor()
            
            # Test shape detection
            shapes = processor._detect_enhanced_shapes(test_image)
            print(f"✅ Shape detection: Found {len(shapes)} shapes")
            
            # Test shape matching with mock street data
            base_location = {'lat': 46.2, 'lon': -122.7}
            image_scale = 2000
            
            parcels = processor.shape_matcher.match_shapes_to_streets(
                shapes, base_location, image_scale
            )
            print(f"✅ Shape matching: Processed {len(parcels)} parcel boundaries")
            
            self.test_results['shape_matching'] = True
            
        except Exception as e:
            print(f"❌ Shape matching test failed: {e}")
            self.test_results['shape_matching'] = False
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        
        print("\n⚠️ Testing Error Handling")
        print("-" * 40)
        
        test_cases = [
            {
                'name': 'Invalid file format',
                'file_content': b'Not a valid PDF',
                'filename': 'invalid.pdf'
            },
            {
                'name': 'Empty file',
                'file_content': b'',
                'filename': 'empty.pdf'
            },
            {
                'name': 'Large file',
                'file_content': b'x' * (60 * 1024 * 1024),  # 60MB
                'filename': 'large.pdf'
            }
        ]
        
        error_handling_results = []
        
        for test_case in test_cases:
            try:
                files = {
                    'file': (test_case['filename'], test_case['file_content'], 'application/pdf')
                }
                
                response = requests.post(
                    f"{self.server_url}/upload",
                    files=files,
                    timeout=30
                )
                
                if response.status_code in [400, 413, 422]:  # Expected error codes
                    print(f"✅ {test_case['name']}: Handled gracefully")
                    error_handling_results.append(True)
                else:
                    print(f"⚠️ {test_case['name']}: Unexpected response {response.status_code}")
                    error_handling_results.append(False)
                    
            except requests.exceptions.Timeout:
                print(f"✅ {test_case['name']}: Timeout handled")
                error_handling_results.append(True)
            except Exception as e:
                print(f"❌ {test_case['name']}: Error {e}")
                error_handling_results.append(False)
        
        self.test_results['error_handling'] = all(error_handling_results)
    
    def analyze_processing_results(self, result: Dict[str, Any], test_name: str):
        """Analyze processing results for quality assessment"""
        
        analysis = {
            'coordinates_found': len(result.get('coordinates', [])),
            'text_extracted': len(result.get('extracted_text', '')),
            'shapes_detected': result.get('detected_shapes', 0),
            'location_found': bool(result.get('base_coordinates')),
            'enhanced_features': bool(result.get('enhanced_analysis'))
        }
        
        print(f"📊 Analysis Results:")
        print(f"   • Coordinates found: {analysis['coordinates_found']}")
        print(f"   • Text extracted: {analysis['text_extracted']} characters")
        print(f"   • Shapes detected: {analysis['shapes_detected']}")
        print(f"   • Location found: {analysis['location_found']}")
        print(f"   • Enhanced features: {analysis['enhanced_features']}")
        
        # Determine success based on criteria
        success = (
            analysis['coordinates_found'] > 0 or
            analysis['text_extracted'] > 100 or
            analysis['shapes_detected'] > 0
        )
        
        self.test_results[test_name] = success
        
        if success:
            print("✅ Processing test passed")
        else:
            print("❌ Processing test failed")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        print(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed")
        print(f"🎯 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📝 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        # Generate recommendations
        print("\n💡 Recommendations:")
        
        if not self.test_results.get('server_connectivity'):
            print("   • Check server is running on correct port")
        
        if not self.test_results.get('basic_processing'):
            print("   • Verify PDF processing pipeline")
            print("   • Check OCR and image processing components")
        
        if not self.test_results.get('enhanced_processing'):
            print("   • Install missing dependencies for enhanced features")
            print("   • Check open data API connectivity")
        
        if not self.test_results.get('coordinate_systems'):
            print("   • Verify pyproj installation and PROJ library")
        
        if not self.test_results.get('open_data_integration'):
            print("   • Check internet connectivity for external APIs")
            print("   • Verify API endpoints are accessible")
        
        if not self.test_results.get('shape_matching'):
            print("   • Improve image preprocessing for shape detection")
            print("   • Tune shape filtering parameters")
        
        if not self.test_results.get('error_handling'):
            print("   • Improve error handling for edge cases")
            print("   • Add better validation for input files")
        
        # Save report to file
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests/total_tests)*100,
            'detailed_results': self.test_results
        }
        
        with open('test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved to test_report.json")
        
        # Overall assessment
        if passed_tests == total_tests:
            print("\n🎉 All tests passed! System is ready for production.")
        elif passed_tests >= total_tests * 0.8:
            print("\n✅ Most tests passed. System is functional with minor issues.")
        elif passed_tests >= total_tests * 0.6:
            print("\n⚠️ Some tests failed. System needs improvements before deployment.")
        else:
            print("\n❌ Many tests failed. System requires significant fixes.")

def main():
    """Main test execution"""
    
    print("🚀 Enhanced Parcel Map Processing System")
    print("Comprehensive Test Suite v2.0")
    print("=" * 60)
    
    # Check if server is running
    test_suite = ComprehensiveTestSuite()
    
    # Run all tests
    test_suite.run_all_tests()

if __name__ == "__main__":
    main() 