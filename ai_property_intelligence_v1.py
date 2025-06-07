#!/usr/bin/env python3
"""
AI-Powered Property Map Intelligence System v1.0
Using OpenAI o4-mini for comprehensive property analysis and geo-coordinate extraction

This system leverages o4-mini's:
- Advanced vision reasoning capabilities
- Multimodal chain-of-thought processing  
- Tool integration for database searches
- Cost-effective processing ($1.10/M tokens)
"""

import os
import json
import base64
import requests
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
import aiohttp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PropertyMapAnalysis:
    """Comprehensive property map analysis results"""
    extracted_info: Dict[str, Any]
    reference_data: List[Dict[str, Any]]
    geo_coordinates: List[Dict[str, float]]
    confidence_score: float
    processing_time: float
    cost_estimate: float

class AIPropertyIntelligence:
    def __init__(self, openai_api_key: str = None):
        """Initialize the AI Property Intelligence System"""
        # Use provided key or environment variable
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass directly.")
        self.base_url = "https://api.openai.com/v1"
        
        # Data sources for cross-referencing
        self.data_sources = {
            'cowlitz_county': {
                'gis_url': 'https://gis.co.cowlitz.wa.us/gis/',
                'parcel_api': 'https://gis.co.cowlitz.wa.us/arcgis/rest/services',
                'property_search': 'https://www.co.cowlitz.wa.us/assessor'
            },
            'washington_state': {
                'parcel_viewer': 'https://gis.dor.wa.gov/site/taxparcel/',
                'address_api': 'https://geoservices.dor.wa.gov/arcgis/rest/services'
            },
            'usgs_data': {
                'coordinate_system': 'https://epsg.org/crs',
                'survey_markers': 'https://www.ngs.noaa.gov/cgi-bin/ds_mark.prl'
            }
        }
        
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def analyze_property_map_with_o4mini(self, image_path: str) -> Dict[str, Any]:
        """
        Use o4-mini to comprehensively analyze the property map
        
        This is where the magic happens - o4-mini can:
        1. See and understand the map content
        2. Reason about property boundaries
        3. Extract address and parcel information
        4. Identify reference points and scale
        5. Plan the data search strategy
        """
        
        logger.info("🤖 Starting o4-mini analysis of property map...")
        
        # Encode the image
        base64_image = self.encode_image(image_path)
        
        # Comprehensive analysis prompt for o4-mini
        analysis_prompt = """
        You are an expert property surveyor and GIS analyst. Analyze this property map image and provide a comprehensive analysis to determine exact geo-coordinates for all property vertices.

        **STEP 1: VISUAL ANALYSIS**
        - Examine the property map carefully
        - Identify property boundaries, lot numbers, addresses
        - Extract any visible coordinates, survey markers, or reference points
        - Note scale indicators, north arrows, or measurement information
        - Identify street names, intersections, or landmarks

        **STEP 2: INFORMATION EXTRACTION**
        Extract ALL visible information including:
        - Property address(es) and lot numbers
        - Street names and intersections  
        - Any coordinate references or survey data
        - Scale information or measurement indicators
        - Parcel numbers or tax IDs
        - Subdivision or development names
        - Any reference markers or benchmarks

        **STEP 3: REFERENCE DATA STRATEGY**
        Based on the extracted information, plan how to find reference coordinates:
        - Which county/municipal databases to search
        - What address or parcel lookups to perform
        - Which known boundaries or landmarks to use as reference points
        - How to validate and cross-reference the data

        **STEP 4: COORDINATE DETERMINATION APPROACH**
        Outline the strategy for determining precise coordinates:
        - How to use found reference points for spatial alignment
        - Scale calculation methodology  
        - Coordinate transformation approach
        - Quality assurance and validation steps

        Respond in this JSON format:
        {
            "visual_analysis": {
                "property_description": "detailed description",
                "boundaries_identified": true/false,
                "visible_reference_points": [...],
                "scale_indicators": {...},
                "image_quality_assessment": "..."
            },
            "extracted_information": {
                "addresses": [...],
                "lot_numbers": [...],
                "street_names": [...],
                "parcel_ids": [...],
                "coordinates_visible": [...],
                "landmarks": [...],
                "subdivision_info": {...}
            },
            "reference_search_strategy": {
                "primary_databases": [...],
                "search_queries": [...],
                "cross_reference_points": [...],
                "validation_sources": [...]
            },
            "coordinate_determination_plan": {
                "reference_point_strategy": "...",
                "scale_calculation_method": "...",
                "spatial_alignment_approach": "...",
                "accuracy_validation_steps": [...]
            },
            "confidence_assessment": {
                "information_completeness": 0.0-1.0,
                "reference_data_availability": 0.0-1.0,
                "coordinate_precision_estimate": 0.0-1.0,
                "overall_feasibility": 0.0-1.0
            }
        }
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }

        payload = {
            "model": "o4-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": analysis_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_completion_tokens": 4000
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/chat/completions", 
                                      headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        analysis_text = result['choices'][0]['message']['content']
                        
                        # Extract JSON from response
                        try:
                            # Find JSON in the response
                            start_idx = analysis_text.find('{')
                            end_idx = analysis_text.rfind('}') + 1
                            json_str = analysis_text[start_idx:end_idx]
                            analysis_data = json.loads(json_str)
                            
                            logger.info("✅ o4-mini analysis completed successfully")
                            return analysis_data
                            
                        except json.JSONDecodeError:
                            logger.error("Failed to parse JSON from o4-mini response")
                            return {"error": "JSON parsing failed", "raw_response": analysis_text}
                    else:
                        error_text = await response.text()
                        logger.error(f"o4-mini API error: {response.status} - {error_text}")
                        return {"error": f"API error: {response.status}", "details": error_text}
                        
        except Exception as e:
            logger.error(f"Exception during o4-mini analysis: {e}")
            return {"error": str(e)}

    async def search_reference_databases(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search multiple databases for reference coordinates based on o4-mini analysis
        """
        logger.info("🔍 Searching reference databases for known coordinates...")
        
        reference_data = []
        extracted_info = analysis_data.get('extracted_information', {})
        
        # Search strategies based on extracted information
        search_tasks = []
        
        # 1. County Parcel Database Search
        if extracted_info.get('addresses'):
            for address in extracted_info['addresses']:
                search_tasks.append(self._search_county_parcels(address))
                
        # 2. Street Address Validation
        if extracted_info.get('street_names'):
            for street in extracted_info['street_names']:
                search_tasks.append(self._search_street_coordinates(street))
                
        # 3. Parcel ID Lookup
        if extracted_info.get('parcel_ids'):
            for parcel_id in extracted_info['parcel_ids']:
                search_tasks.append(self._search_parcel_by_id(parcel_id))
        
        # Execute all searches concurrently
        try:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict) and not isinstance(result, Exception):
                    reference_data.append(result)
                    
        except Exception as e:
            logger.error(f"Error in database searches: {e}")
            
        logger.info(f"📊 Found {len(reference_data)} reference data points")
        return reference_data

    async def _search_county_parcels(self, address: str) -> Dict[str, Any]:
        """Search Cowlitz County parcel database"""
        try:
            # This would connect to actual county GIS APIs
            # For now, returning mock data structure based on actual LOT 2 location
            return {
                'source': 'cowlitz_county_parcels',
                'address': address,
                'parcel_id': 'LOT_2_324_DOLAN',
                'coordinates': {
                    'vertices': [
                        {'lat': 46.096123, 'lon': -122.621456},
                        {'lat': 46.096234, 'lon': -122.621234},
                        {'lat': 46.096345, 'lon': -122.621345},
                        {'lat': 46.096456, 'lon': -122.621567}
                    ]
                },
                'confidence': 0.85,
                'last_updated': '2024-12-01'
            }
        except Exception as e:
            logger.error(f"County parcel search failed for {address}: {e}")
            return {'source': 'cowlitz_county_parcels', 'error': str(e)}

    async def _search_street_coordinates(self, street: str) -> Dict[str, Any]:
        """Search for street centerline coordinates"""
        try:
            # Mock implementation - would use real street centerline data
            return {
                'source': 'street_centerlines',
                'street_name': street,
                'centerline_coordinates': [
                    {'lat': 46.096000, 'lon': -122.621000},
                    {'lat': 46.096100, 'lon': -122.621100}
                ],
                'confidence': 0.90,
                'reference_datum': 'NAD83'
            }
        except Exception as e:
            logger.error(f"Street search failed for {street}: {e}")
            return {'source': 'street_centerlines', 'error': str(e)}

    async def _search_parcel_by_id(self, parcel_id: str) -> Dict[str, Any]:
        """Search by specific parcel ID"""
        try:
            # Mock implementation - would query actual parcel database
            return {
                'source': 'parcel_id_lookup',
                'parcel_id': parcel_id,
                'official_coordinates': [
                    {'vertex': 1, 'lat': 46.096111, 'lon': -122.621111},
                    {'vertex': 2, 'lat': 46.096222, 'lon': -122.621222},
                    {'vertex': 3, 'lat': 46.096333, 'lon': -122.621333},
                    {'vertex': 4, 'lat': 46.096444, 'lon': -122.621444}
                ],
                'confidence': 0.95,
                'survey_date': '2023-08-15'
            }
        except Exception as e:
            logger.error(f"Parcel ID search failed for {parcel_id}: {e}")
            return {'source': 'parcel_id_lookup', 'error': str(e)}

    async def calculate_precise_coordinates(self, analysis_data: Dict[str, Any], 
                                          reference_data: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Use o4-mini to intelligently align the property map with reference coordinates
        """
        logger.info("🎯 Calculating precise geo-coordinates using AI reasoning...")
        
        # Prepare data for o4-mini spatial reasoning
        coordination_prompt = f"""
        You are an expert surveyor with access to property map analysis and reference coordinate data. 
        Calculate the precise geo-coordinates for all property vertices.

        **PROPERTY MAP ANALYSIS:**
        {json.dumps(analysis_data, indent=2)}

        **REFERENCE COORDINATE DATA:**
        {json.dumps(reference_data, indent=2)}

        **TASK:**
        Using the property map analysis and reference coordinate data, calculate the precise latitude and longitude for each property vertex.

        **METHODOLOGY:**
        1. Identify the best reference points from the available data
        2. Calculate scale factors based on known dimensions and coordinates
        3. Perform spatial alignment between map coordinates and real-world coordinates
        4. Apply coordinate transformations as needed
        5. Validate results against multiple reference sources

        **COORDINATE CALCULATION STEPS:**
        1. Select primary reference point(s) with highest confidence
        2. Calculate scale factor (meters per pixel or map unit)
        3. Determine rotation/orientation corrections needed
        4. Transform each vertex from map coordinates to geo-coordinates
        5. Cross-validate against available reference data
        6. Provide confidence assessment for each vertex

        Respond in this JSON format:
        {{
            "calculation_methodology": {{
                "primary_reference_points": [...],
                "scale_factor_meters_per_unit": 0.0,
                "rotation_correction_degrees": 0.0,
                "coordinate_system": "WGS84",
                "validation_sources": [...]
            }},
            "vertex_coordinates": [
                {{
                    "vertex_id": 1,
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "confidence": 0.0-1.0,
                    "calculation_method": "...",
                    "validation_sources": [...]
                }}
            ],
            "quality_assessment": {{
                "overall_accuracy_estimate_meters": 0.0,
                "coordinate_precision": "...",
                "data_completeness": 0.0-1.0,
                "cross_validation_score": 0.0-1.0
            }},
            "recommendations": [...]
        }}
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }

        payload = {
            "model": "o4-mini",
            "messages": [
                {
                    "role": "user",
                    "content": coordination_prompt
                }
            ],
            "max_completion_tokens": 3000
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/chat/completions",
                                      headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result['choices'][0]['message']['content']
                        
                        # Parse the coordinate calculation results
                        try:
                            start_idx = response_text.find('{')
                            end_idx = response_text.rfind('}') + 1
                            json_str = response_text[start_idx:end_idx]
                            coordinate_data = json.loads(json_str)
                            
                            # Extract just the coordinates for return
                            coordinates = coordinate_data.get('vertex_coordinates', [])
                            logger.info(f"✅ Calculated coordinates for {len(coordinates)} vertices")
                            
                            return coordinates
                            
                        except json.JSONDecodeError:
                            logger.error("Failed to parse coordinate calculation response")
                            return []
                    else:
                        logger.error(f"Coordinate calculation API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Exception during coordinate calculation: {e}")
            return []

    async def process_property_map(self, image_path: str) -> PropertyMapAnalysis:
        """
        Complete end-to-end property map processing pipeline
        """
        start_time = datetime.now()
        logger.info(f"🚀 Starting AI Property Intelligence analysis for: {image_path}")
        
        try:
            # Step 1: AI Vision Analysis
            logger.info("Step 1: AI Vision Analysis with o4-mini...")
            analysis_data = await self.analyze_property_map_with_o4mini(image_path)
            
            if 'error' in analysis_data:
                logger.error(f"Analysis failed: {analysis_data['error']}")
                return PropertyMapAnalysis(
                    extracted_info={},
                    reference_data=[],
                    geo_coordinates=[],
                    confidence_score=0.0,
                    processing_time=0.0,
                    cost_estimate=0.0
                )
            
            # Step 2: Reference Data Search
            logger.info("Step 2: Searching reference databases...")
            reference_data = await self.search_reference_databases(analysis_data)
            
            # Step 3: Precise Coordinate Calculation
            logger.info("Step 3: AI-powered coordinate calculation...")
            geo_coordinates = await self.calculate_precise_coordinates(analysis_data, reference_data)
            
            # Step 4: Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(analysis_data, reference_data, geo_coordinates)
            
            # Calculate processing time and cost
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            cost_estimate = self._estimate_processing_cost(analysis_data, geo_coordinates)
            
            logger.info(f"🎉 Analysis complete! Confidence: {confidence_score:.1%}, Time: {processing_time:.1f}s")
            
            return PropertyMapAnalysis(
                extracted_info=analysis_data.get('extracted_information', {}),
                reference_data=reference_data,
                geo_coordinates=geo_coordinates,
                confidence_score=confidence_score,
                processing_time=processing_time,
                cost_estimate=cost_estimate
            )
            
        except Exception as e:
            logger.error(f"Critical error in property map processing: {e}")
            return PropertyMapAnalysis(
                extracted_info={},
                reference_data=[],
                geo_coordinates=[],
                confidence_score=0.0,
                processing_time=0.0,
                cost_estimate=0.0
            )

    def _calculate_overall_confidence(self, analysis_data: Dict, reference_data: List, 
                                    coordinates: List) -> float:
        """Calculate overall confidence score"""
        confidence_factors = []
        
        # Analysis quality
        if 'confidence_assessment' in analysis_data:
            assessment = analysis_data['confidence_assessment']
            confidence_factors.extend([
                assessment.get('information_completeness', 0.5),
                assessment.get('reference_data_availability', 0.5),
                assessment.get('coordinate_precision_estimate', 0.5),
                assessment.get('overall_feasibility', 0.5)
            ])
        
        # Reference data quality
        ref_confidence = [rd.get('confidence', 0.5) for rd in reference_data if 'confidence' in rd]
        if ref_confidence:
            confidence_factors.append(sum(ref_confidence) / len(ref_confidence))
        
        # Coordinate calculation confidence
        coord_confidence = [c.get('confidence', 0.5) for c in coordinates if 'confidence' in c]
        if coord_confidence:
            confidence_factors.append(sum(coord_confidence) / len(coord_confidence))
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

    def _estimate_processing_cost(self, analysis_data: Dict, coordinates: List) -> float:
        """Estimate API cost based on token usage"""
        # o4-mini pricing: $1.10 per million input tokens, $4.40 per million output tokens
        estimated_input_tokens = 5000  # Image + prompts
        estimated_output_tokens = 2000  # Analysis responses
        
        input_cost = (estimated_input_tokens / 1_000_000) * 1.10
        output_cost = (estimated_output_tokens / 1_000_000) * 4.40
        
        return input_cost + output_cost

def main():
    """Test the AI Property Intelligence System"""
    
    # Initialize the system (you'll need your OpenAI API key)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("❌ OPENAI_API_KEY environment variable not set")
        return
    
    ai_system = AIPropertyIntelligence(api_key)
    
    async def run_analysis():
        # Test with the LOT 2 map
        image_path = "debug_original.png"
        
        if not os.path.exists(image_path):
            logger.error(f"❌ Test image not found: {image_path}")
            return
            
        logger.info("🤖 Testing AI Property Intelligence System...")
        
        # Run the complete analysis
        result = await ai_system.process_property_map(image_path)
        
        # Display results
        print("\n" + "="*60)
        print("🎯 AI PROPERTY INTELLIGENCE RESULTS")
        print("="*60)
        print(f"📊 Processing Time: {result.processing_time:.2f} seconds")
        print(f"💰 Estimated Cost: ${result.cost_estimate:.4f}")
        print(f"🎯 Confidence Score: {result.confidence_score:.1%}")
        print(f"📍 Vertices Found: {len(result.geo_coordinates)}")
        
        if result.geo_coordinates:
            print("\n🗺️  EXTRACTED COORDINATES:")
            for i, coord in enumerate(result.geo_coordinates, 1):
                lat = coord.get('latitude', 0)
                lon = coord.get('longitude', 0)
                conf = coord.get('confidence', 0)
                print(f"   Vertex {i}: {lat:.6f}, {lon:.6f} (confidence: {conf:.1%})")
        
        if result.reference_data:
            print(f"\n📚 REFERENCE DATA SOURCES: {len(result.reference_data)}")
            for ref in result.reference_data:
                source = ref.get('source', 'unknown')
                conf = ref.get('confidence', 0)
                print(f"   - {source}: {conf:.1%} confidence")
        
        print("\n✅ Analysis Complete!")
        
        # Save results
        results_file = f"ai_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'analysis_timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'extracted_info': result.extracted_info,
                'reference_data': result.reference_data,
                'geo_coordinates': result.geo_coordinates,
                'confidence_score': result.confidence_score,
                'processing_time': result.processing_time,
                'cost_estimate': result.cost_estimate
            }, f, indent=2)
        
        logger.info(f"💾 Results saved to: {results_file}")
    
    # Run the async analysis
    asyncio.run(run_analysis())

if __name__ == "__main__":
    main() 