#!/usr/bin/env python3
"""
Enhanced Geocoding for Parcel Maps
Improved location detection and coordinate resolution with multiple fallback strategies
"""

import re
import requests
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import json

logger = logging.getLogger(__name__)

class EnhancedGeocoder:
    """Enhanced geocoding with multiple strategies and location context"""
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="Enhanced-Parcel-Processor-v2.0")
        self.cache = {}
        
    def analyze_location_context(self, extracted_text: str) -> Dict[str, Any]:
        """Enhanced location analysis with better pattern matching"""
        
        logger.info("🌍 Starting enhanced location analysis...")
        
        location_info = {
            'primary_address': None,
            'streets': [],
            'county': None,
            'state': None,
            'zip_code': None,
            'coordinates': [],
            'lot_info': [],
            'confidence_score': 0.0
        }
        
        # Enhanced address patterns
        location_info['primary_address'] = self._extract_primary_address(extracted_text)
        location_info['streets'] = self._extract_streets_enhanced(extracted_text)
        location_info['county'] = self._extract_county_enhanced(extracted_text)
        location_info['state'] = self._extract_state_enhanced(extracted_text)
        location_info['zip_code'] = self._extract_zip_code(extracted_text)
        location_info['coordinates'] = self._extract_coordinate_references(extracted_text)
        location_info['lot_info'] = self._extract_lot_info(extracted_text)
        
        # Calculate confidence score
        location_info['confidence_score'] = self._calculate_location_confidence(location_info)
        
        logger.info(f"📍 Location analysis complete - Confidence: {location_info['confidence_score']:.2f}")
        return location_info
    
    def _extract_primary_address(self, text: str) -> Optional[str]:
        """Extract the primary address from text"""
        
        # Patterns for addresses
        address_patterns = [
            r'\(LOT\s+\d+\)\s+(\d+\s+[A-Z][A-Za-z\s]+(?:Road|Rd|Street|St|Avenue|Ave|Drive|Dr|Way|Lane|Ln))',
            r'(\d+\s+[A-Z][A-Za-z\s]+(?:Road|Rd|Street|St|Avenue|Ave|Drive|Dr|Way|Lane|Ln))',
            r'(\d+\s+[A-Z][A-Za-z\s]+\s+(?:RD|ROAD|ST|STREET|AVE|AVENUE|DR|DRIVE|WAY|LN|LANE))',
        ]
        
        for pattern in address_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Return the first (most likely primary) address
                address = matches[0].strip()
                logger.info(f"🏠 Primary address found: '{address}'")
                return address
        
        return None
    
    def _extract_streets_enhanced(self, text: str) -> List[str]:
        """Enhanced street extraction with better filtering"""
        
        street_patterns = [
            r'(\d+\s+[A-Z][A-Za-z\s]+(?:ROAD|RD|STREET|ST|AVENUE|AVE|DRIVE|DR|WAY|LANE|LN|BOULEVARD|BLVD))',
            r'([A-Z][A-Za-z\s]{2,30}\s+(?:ROAD|RD|STREET|ST|AVENUE|AVE|DRIVE|DR|WAY|LANE|LN|BOULEVARD|BLVD))',
            r'([A-Z][A-Za-z\s]+\s+(?:RD|ROAD|ST|STREET|AVE|AVENUE|DR|DRIVE|WAY|LN|LANE))',
        ]
        
        streets = set()
        
        for pattern in street_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cleaned = self._clean_street_name(match)
                if cleaned and self._is_valid_street_name(cleaned):
                    streets.add(cleaned)
        
        street_list = list(streets)
        logger.info(f"🛣️ Found {len(street_list)} unique streets")
        return street_list
    
    def _clean_street_name(self, street: str) -> str:
        """Clean and normalize street name"""
        
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', street.strip())
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Standardize common abbreviations
        replacements = {
            ' RD': ' ROAD',
            ' ST': ' STREET', 
            ' AVE': ' AVENUE',
            ' DR': ' DRIVE',
            ' LN': ' LANE',
            ' BLVD': ' BOULEVARD'
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned.strip()
    
    def _is_valid_street_name(self, street: str) -> bool:
        """Check if street name is valid (not OCR artifact)"""
        
        # Too short or too long
        if len(street) < 5 or len(street) > 50:
            return False
        
        # Must contain a street suffix
        suffixes = ['ROAD', 'STREET', 'AVENUE', 'DRIVE', 'WAY', 'LANE', 'BOULEVARD']
        if not any(suffix in street.upper() for suffix in suffixes):
            return False
        
        # Must not be mostly numbers or special characters
        letter_count = sum(1 for c in street if c.isalpha())
        if letter_count < len(street) * 0.5:
            return False
        
        # Filter out common OCR artifacts
        artifacts = ['DATA STREET', 'EPIC STREET', 'POINTS STREET', 'ADDRESS STREET']
        if any(artifact in street.upper() for artifact in artifacts):
            return False
        
        return True
    
    def _extract_county_enhanced(self, text: str) -> Optional[str]:
        """Enhanced county extraction with better patterns"""
        
        county_patterns = [
            r'([A-Z][A-Za-z\s]+)\s+County(?:\s+GIS)?',
            r'County\s+of\s+([A-Z][A-Za-z\s]+)',
            r'([A-Z][A-Za-z]+)\s+Co\.?(?:\s|$)',
        ]
        
        for pattern in county_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                county = match.strip()
                if len(county) > 2 and county.upper() not in ['GIS', 'DATA', 'DEPARTMENT']:
                    logger.info(f"🏛️ County found: '{county} County'")
                    return f"{county} County"
        
        return None
    
    def _extract_state_enhanced(self, text: str) -> Optional[str]:
        """Enhanced state extraction"""
        
        # Common state patterns
        state_patterns = [
            r'\b(WA|Washington)\b',
            r'\b(OR|Oregon)\b', 
            r'\b(CA|California)\b',
            r'\b(ID|Idaho)\b',
            r'\b(MT|Montana)\b',
            r'Washington\s+(?:State|County)',
        ]
        
        for pattern in state_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                state = matches[0].upper()
                if state == 'WASHINGTON':
                    state = 'WA'
                logger.info(f"🗺️ State found: '{state}'")
                return state
        
        return None
    
    def _extract_zip_code(self, text: str) -> Optional[str]:
        """Extract ZIP code from text"""
        
        zip_patterns = [
            r'\b(\d{5})\b',
            r'\b(\d{5}-\d{4})\b'
        ]
        
        for pattern in zip_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Validate ZIP code range (US ZIP codes)
                if match.isdigit() and 10000 <= int(match) <= 99999:
                    logger.info(f"📮 ZIP code found: '{match}'")
                    return match
        
        return None
    
    def _extract_coordinate_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract coordinate references from text"""
        
        coordinates = []
        
        # Decimal degrees
        decimal_pattern = r'(\-?\d{1,3}\.\d{4,8})[,\s]+(\-?\d{1,3}\.\d{4,8})'
        matches = re.findall(decimal_pattern, text)
        
        for lat_str, lon_str in matches:
            try:
                lat, lon = float(lat_str), float(lon_str)
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    coordinates.append({
                        'type': 'decimal_degrees',
                        'latitude': lat,
                        'longitude': lon,
                        'source': 'extracted'
                    })
            except ValueError:
                continue
        
        # DMS (Degrees, Minutes, Seconds) - if found in text
        dms_pattern = r'(\d{1,3})[°\s]+(\d{1,2})[′\'\s]+(\d{1,2}(?:\.\d+)?)[″\"\s]*([NSEW])'
        dms_matches = re.findall(dms_pattern, text)
        
        if len(dms_matches) >= 2:
            # Try to pair latitude and longitude
            try:
                lat_dms = dms_matches[0]
                lon_dms = dms_matches[1]
                
                lat = self._dms_to_decimal(lat_dms)
                lon = self._dms_to_decimal(lon_dms)
                
                if lat and lon:
                    coordinates.append({
                        'type': 'dms',
                        'latitude': lat,
                        'longitude': lon,
                        'source': 'extracted'
                    })
            except:
                pass
        
        if coordinates:
            logger.info(f"📊 Found {len(coordinates)} coordinate references")
        
        return coordinates
    
    def _dms_to_decimal(self, dms_tuple: Tuple[str, str, str, str]) -> Optional[float]:
        """Convert DMS to decimal degrees"""
        
        try:
            degrees, minutes, seconds, direction = dms_tuple
            decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
            
            if direction.upper() in ['S', 'W']:
                decimal = -decimal
            
            return decimal
        except:
            return None
    
    def _extract_lot_info(self, text: str) -> List[str]:
        """Extract lot and parcel information"""
        
        lot_patterns = [
            r'LOT\s+(\d+[A-Z]?)',
            r'PARCEL\s+(\w+)',
            r'BLOCK\s+(\d+)',
            r'TRACT\s+(\w+)',
        ]
        
        lots = []
        for pattern in lot_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            lots.extend(matches)
        
        if lots:
            logger.info(f"📦 Found lot info: {lots}")
        
        return lots
    
    def _calculate_location_confidence(self, location_info: Dict[str, Any]) -> float:
        """Calculate confidence score for location information"""
        
        score = 0.0
        
        # Primary address (30%)
        if location_info['primary_address']:
            score += 0.30
        
        # County (20%)
        if location_info['county']:
            score += 0.20
        
        # State (20%)
        if location_info['state']:
            score += 0.20
        
        # Streets (15%)
        if location_info['streets']:
            score += min(0.15, len(location_info['streets']) * 0.05)
        
        # ZIP code (10%)
        if location_info['zip_code']:
            score += 0.10
        
        # Coordinates (5%)
        if location_info['coordinates']:
            score += 0.05
        
        return min(score, 1.0)
    
    def geocode_with_context(self, location_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced geocoding with context and fallback strategies"""
        
        logger.info("🔍 Starting enhanced geocoding with context...")
        
        # Strategy 1: Primary address + location context
        if location_info.get('primary_address'):
            result = self._geocode_primary_address(location_info)
            if result:
                return result
        
        # Strategy 2: Street + county + state
        if location_info.get('streets') and location_info.get('county'):
            result = self._geocode_street_context(location_info)
            if result:
                return result
        
        # Strategy 3: County + state only (fallback)
        if location_info.get('county') and location_info.get('state'):
            result = self._geocode_area_context(location_info)
            if result:
                return result
        
        # Strategy 4: Use extracted coordinates if available
        if location_info.get('coordinates'):
            coord_ref = location_info['coordinates'][0]
            return {
                'latitude': coord_ref['latitude'],
                'longitude': coord_ref['longitude'],
                'address': 'Extracted from document coordinates',
                'confidence': 0.7,
                'source': 'coordinate_extraction'
            }
        
        logger.warning("❌ All geocoding strategies failed")
        return None
    
    def _geocode_primary_address(self, location_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Geocode using primary address with location context"""
        
        address = location_info['primary_address']
        county = location_info.get('county', '')
        state = location_info.get('state', '')
        
        # Build query variations
        queries = []
        
        if county and state:
            queries.append(f"{address}, {county}, {state}")
        if state:
            queries.append(f"{address}, {state}")
        queries.append(address)
        
        for query in queries:
            logger.info(f"🔍 Geocoding query: '{query}'")
            result = self._geocode_query(query)
            if result:
                # Validate result makes sense for the context
                if self._validate_geocoding_result(result, location_info):
                    result['confidence'] = 0.9
                    result['source'] = 'primary_address'
                    return result
        
        return None
    
    def _geocode_street_context(self, location_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Geocode using street and area context"""
        
        streets = location_info['streets']
        county = location_info.get('county', '')
        state = location_info.get('state', '')
        
        # Try with the first (most likely) street
        if streets:
            street = streets[0]
            
            queries = []
            if county and state:
                queries.append(f"{street}, {county}, {state}")
            if state:
                queries.append(f"{street}, {state}")
            
            for query in queries:
                logger.info(f"🔍 Street geocoding query: '{query}'")
                result = self._geocode_query(query)
                if result:
                    if self._validate_geocoding_result(result, location_info):
                        result['confidence'] = 0.7
                        result['source'] = 'street_context'
                        return result
        
        return None
    
    def _geocode_area_context(self, location_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Geocode using area context only (fallback)"""
        
        county = location_info.get('county', '')
        state = location_info.get('state', '')
        
        if county and state:
            query = f"{county}, {state}"
            logger.info(f"🔍 Area geocoding query: '{query}'")
            result = self._geocode_query(query)
            if result:
                result['confidence'] = 0.5
                result['source'] = 'area_context'
                return result
        
        return None
    
    def _geocode_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Execute geocoding query with error handling"""
        
        # Check cache first
        if query in self.cache:
            logger.debug(f"📋 Using cached result for: '{query}'")
            return self.cache[query]
        
        try:
            location = self.geolocator.geocode(query, timeout=10)
            if location:
                result = {
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                    'address': location.address,
                    'query': query
                }
                
                # Cache the result
                self.cache[query] = result
                
                logger.info(f"✅ Geocoding successful: {location.latitude}, {location.longitude}")
                return result
            else:
                logger.warning(f"⚠️ No results for query: '{query}'")
                return None
                
        except GeocoderTimedOut:
            logger.warning(f"⏰ Geocoding timeout for query: '{query}'")
            return None
        except Exception as e:
            logger.error(f"❌ Geocoding error for query '{query}': {e}")
            return None
    
    def _validate_geocoding_result(self, result: Dict[str, Any], location_info: Dict[str, Any]) -> bool:
        """Validate that geocoding result makes sense for the context"""
        
        lat, lon = result['latitude'], result['longitude']
        
        # Basic coordinate validation
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return False
        
        # State-specific validation
        state = location_info.get('state')
        if state == 'WA':
            # Washington state bounds (approximate)
            if not (45.5 <= lat <= 49.0 and -124.8 <= lon <= -116.9):
                logger.warning(f"⚠️ Coordinates {lat}, {lon} outside Washington state bounds")
                return False
        
        # County-specific validation could be added here
        
        return True

def create_enhanced_geocoder() -> EnhancedGeocoder:
    """Factory function to create enhanced geocoder"""
    return EnhancedGeocoder() 