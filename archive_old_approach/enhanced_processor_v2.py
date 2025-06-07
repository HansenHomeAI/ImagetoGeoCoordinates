#!/usr/bin/env python3
"""
Enhanced Parcel Map Processor v2.0
- Integrates with free open source parcel data
- Advanced coordinate system detection and conversion
- Shape matching with street networks
- Robust property boundary detection
"""

import cv2
import numpy as np
import requests
import json
import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pyproj
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import transform
import matplotlib.pyplot as plt
from functools import partial

@dataclass
class CoordinateSystem:
    """Represents a coordinate system with conversion capabilities"""
    name: str
    epsg_code: int
    proj_string: str
    unit: str
    zone: Optional[str] = None

@dataclass
class ParcelBoundary:
    """Represents a property boundary with metadata"""
    vertices: List[Tuple[float, float]]
    area_sqft: float
    perimeter_ft: float
    confidence: float
    coordinate_system: str
    source: str

@dataclass
class StreetNetwork:
    """Represents street network data for matching"""
    name: str
    geometry: LineString
    classification: str
    width_ft: Optional[float] = None

class OpenDataIntegrator:
    """Integrates with free open source GIS data sources"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ParcelMapProcessor/2.0 (+https://github.com/user/repo)'
        })
    
    def query_overpass_api(self, bbox: Tuple[float, float, float, float], 
                          query_type: str = "highway") -> List[Dict]:
        """Query OpenStreetMap data via Overpass API"""
        
        south, west, north, east = bbox
        
        overpass_query = f"""
        [out:json][timeout:25];
        (
          way["{query_type}"]({south},{west},{north},{east});
          relation["{query_type}"]({south},{west},{north},{east});
        );
        out geom;
        """
        
        overpass_url = "http://overpass-api.de/api/interpreter"
        
        try:
            response = self.session.post(overpass_url, data=overpass_query, timeout=30)
            response.raise_for_status()
            return response.json().get('elements', [])
        except Exception as e:
            self.logger.warning(f"Overpass API query failed: {e}")
            return []
    
    def query_census_tiger(self, bbox: Tuple[float, float, float, float]) -> List[Dict]:
        """Query US Census TIGER/Line data for roads"""
        
        # Census Bureau's REST API for TIGER data
        base_url = "https://tigerweb.geo.census.gov/arcgis/rest/services"
        service_url = f"{base_url}/TIGERweb/Transportation/MapServer/0/query"
        
        params = {
            'where': '1=1',
            'geometry': f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",
            'geometryType': 'esriGeometryEnvelope',
            'spatialRel': 'esriSpatialRelIntersects',
            'outFields': 'FULLNAME,MTFCC,RTTYP',
            'returnGeometry': 'true',
            'f': 'json'
        }
        
        try:
            response = self.session.get(service_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('features', [])
        except Exception as e:
            self.logger.warning(f"Census TIGER query failed: {e}")
            return []
    
    def get_county_parcels(self, county: str, state: str) -> List[Dict]:
        """Query county parcel data from various sources"""
        
        # Try multiple sources for county parcel data
        sources = [
            self._query_county_arcgis(county, state),
            self._query_state_gis(county, state),
            self._query_local_gis(county, state)
        ]
        
        parcels = []
        for source_data in sources:
            if source_data:
                parcels.extend(source_data)
                break  # Use first successful source
        
        return parcels
    
    def _query_county_arcgis(self, county: str, state: str) -> List[Dict]:
        """Query county ArcGIS services (many counties publish open data)"""
        
        # Common county GIS REST service patterns
        potential_urls = [
            f"https://gis.{county.lower()}county{state.lower()}.gov/arcgis/rest/services",
            f"https://maps.{county.lower()}county.gov/arcgis/rest/services",
            f"https://gis.{county.lower()}.{state.lower()}.us/arcgis/rest/services",
            f"https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/{county}_{state}_Parcels"
        ]
        
        for base_url in potential_urls:
            try:
                # Try to find parcel service
                response = self.session.get(f"{base_url}?f=json", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    # Look for parcel-related services
                    for service in data.get('services', []):
                        if any(term in service['name'].lower() for term in ['parcel', 'property', 'tax']):
                            return self._query_arcgis_service(f"{base_url}/{service['name']}/MapServer")
            except:
                continue
        
        return []
    
    def _query_state_gis(self, county: str, state: str) -> List[Dict]:
        """Query state-level GIS services"""
        # Implementation for state GIS services
        return []
    
    def _query_local_gis(self, county: str, state: str) -> List[Dict]:
        """Query local municipality GIS services"""
        # Implementation for local GIS services
        return []
    
    def _query_arcgis_service(self, service_url: str) -> List[Dict]:
        """Query a specific ArcGIS MapServer service"""
        try:
            response = self.session.get(f"{service_url}/0/query", params={
                'where': '1=1',
                'outFields': '*',
                'returnGeometry': 'true',
                'f': 'json',
                'resultRecordCount': 1000
            }, timeout=30)
            
            if response.status_code == 200:
                return response.json().get('features', [])
        except:
            pass
        
        return []

class AdvancedCoordinateProcessor:
    """Advanced coordinate system detection and conversion"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common coordinate systems in the US
        self.coordinate_systems = {
            'WGS84': CoordinateSystem('WGS84', 4326, '+proj=longlat +datum=WGS84', 'degrees'),
            'NAD83': CoordinateSystem('NAD83', 4269, '+proj=longlat +datum=NAD83', 'degrees'),
            'UTM_Zone_10N': CoordinateSystem('UTM Zone 10N', 32610, '+proj=utm +zone=10 +datum=WGS84', 'meters', '10N'),
            'UTM_Zone_11N': CoordinateSystem('UTM Zone 11N', 32611, '+proj=utm +zone=11 +datum=WGS84', 'meters', '11N'),
            'WA_State_Plane_North': CoordinateSystem('Washington State Plane North', 2285, '+proj=lcc +lat_1=48.73333333333333 +lat_2=47.5 +lat_0=47 +lon_0=-120.8333333333333 +x_0=500000 +y_0=0 +datum=NAD83', 'feet', 'WA North'),
            'WA_State_Plane_South': CoordinateSystem('Washington State Plane South', 2286, '+proj=lcc +lat_1=47.33333333333334 +lat_2=45.83333333333334 +lat_0=45.33333333333334 +lon_0=-120.5 +x_0=500000 +y_0=0 +datum=NAD83', 'feet', 'WA South'),
        }
    
    def detect_coordinate_system(self, coordinates: List[Tuple[float, float]], 
                               location_hint: Optional[Dict] = None) -> CoordinateSystem:
        """Detect the most likely coordinate system for given coordinates"""
        
        if not coordinates:
            return self.coordinate_systems['WGS84']
        
        # Analyze coordinate ranges to determine system
        x_coords = [c[0] for c in coordinates]
        y_coords = [c[1] for c in coordinates]
        
        x_range = (min(x_coords), max(x_coords))
        y_range = (min(y_coords), max(y_coords))
        
        # Check if coordinates are in lat/lon range
        if (-180 <= x_range[0] <= 180 and -90 <= y_range[0] <= 90):
            return self.coordinate_systems['WGS84']
        
        # Check for UTM coordinates (6-7 digit easting, 7 digit northing)
        if (100000 <= x_range[0] <= 999999 and 1000000 <= y_range[0] <= 9999999):
            # Determine UTM zone based on location hint
            if location_hint and location_hint.get('lon'):
                lon = location_hint['lon']
                if -126 <= lon <= -120:
                    return self.coordinate_systems['UTM_Zone_10N']
                elif -120 <= lon <= -114:
                    return self.coordinate_systems['UTM_Zone_11N']
        
        # Check for State Plane coordinates (large numbers, typically feet)
        if location_hint and location_hint.get('state') == 'WA':
            if location_hint.get('lat', 0) > 47.5:
                return self.coordinate_systems['WA_State_Plane_North']
            else:
                return self.coordinate_systems['WA_State_Plane_South']
        
        # Default to WGS84
        return self.coordinate_systems['WGS84']
    
    def convert_coordinates(self, coordinates: List[Tuple[float, float]], 
                          from_system: CoordinateSystem, 
                          to_system: CoordinateSystem) -> List[Tuple[float, float]]:
        """Convert coordinates between different systems"""
        
        if from_system.epsg_code == to_system.epsg_code:
            return coordinates
        
        try:
            transformer = pyproj.Transformer.from_crs(
                from_system.epsg_code, 
                to_system.epsg_code, 
                always_xy=True
            )
            
            converted = []
            for x, y in coordinates:
                new_x, new_y = transformer.transform(x, y)
                converted.append((new_x, new_y))
            
            return converted
            
        except Exception as e:
            self.logger.error(f"Coordinate conversion failed: {e}")
            return coordinates

class ShapeMatchingEngine:
    """Advanced shape matching with street networks and parcel data"""
    
    def __init__(self, open_data: OpenDataIntegrator, coord_processor: AdvancedCoordinateProcessor):
        self.open_data = open_data
        self.coord_processor = coord_processor
        self.logger = logging.getLogger(__name__)
    
    def match_shapes_to_streets(self, shapes: List[np.ndarray], 
                               base_location: Dict,
                               image_scale: float) -> List[ParcelBoundary]:
        """Match detected shapes to street networks for validation"""
        
        # Get street network data
        bbox = self._calculate_search_bbox(base_location, image_scale)
        streets = self._get_street_network(bbox)
        
        matched_parcels = []
        
        for shape in shapes:
            # Convert shape to geographic coordinates
            geo_vertices = self._pixels_to_coordinates(shape, base_location, image_scale)
            
            # Create polygon
            if len(geo_vertices) >= 3:
                polygon = Polygon(geo_vertices)
                
                # Calculate metrics
                area_sqft = self._calculate_area_sqft(polygon)
                perimeter_ft = self._calculate_perimeter_ft(polygon)
                
                # Match with street network
                confidence = self._calculate_street_match_confidence(polygon, streets)
                
                parcel = ParcelBoundary(
                    vertices=geo_vertices,
                    area_sqft=area_sqft,
                    perimeter_ft=perimeter_ft,
                    confidence=confidence,
                    coordinate_system='WGS84',
                    source='image_analysis'
                )
                
                matched_parcels.append(parcel)
        
        return matched_parcels
    
    def cross_reference_parcels(self, detected_parcels: List[ParcelBoundary],
                               county: str, state: str) -> List[ParcelBoundary]:
        """Cross-reference with official parcel data"""
        
        # Get official parcel data
        official_parcels = self.open_data.get_county_parcels(county, state)
        
        enhanced_parcels = []
        
        for detected in detected_parcels:
            detected_polygon = Polygon(detected.vertices)
            
            best_match = None
            best_overlap = 0
            
            # Find best matching official parcel
            for official in official_parcels:
                if 'geometry' in official:
                    try:
                        official_polygon = self._parse_parcel_geometry(official['geometry'])
                        overlap = detected_polygon.intersection(official_polygon).area
                        
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match = official
                    except:
                        continue
            
            # Enhance detected parcel with official data
            if best_match and best_overlap > 0.7 * detected_polygon.area:
                detected.confidence = min(1.0, detected.confidence + 0.3)
                # Add official parcel attributes
                if 'attributes' in best_match:
                    attrs = best_match['attributes']
                    detected.source = f"matched_official_{attrs.get('OBJECTID', 'unknown')}"
            
            enhanced_parcels.append(detected)
        
        return enhanced_parcels
    
    def _calculate_search_bbox(self, base_location: Dict, image_scale: float) -> Tuple[float, float, float, float]:
        """Calculate bounding box for data searches"""
        
        lat = base_location.get('lat', 46.2)
        lon = base_location.get('lon', -122.7)
        
        # Estimate map extent based on scale
        # Assume image covers roughly 0.5 mile radius at 1:2000 scale
        scale_factor = image_scale / 2000.0
        radius_degrees = 0.01 * scale_factor  # Rough approximation
        
        return (
            lat - radius_degrees,  # south
            lon - radius_degrees,  # west
            lat + radius_degrees,  # north
            lon + radius_degrees   # east
        )
    
    def _get_street_network(self, bbox: Tuple[float, float, float, float]) -> List[StreetNetwork]:
        """Get street network data for the area"""
        
        streets = []
        
        # Try OpenStreetMap first
        osm_data = self.open_data.query_overpass_api(bbox, "highway")
        for element in osm_data:
            if element.get('type') == 'way' and 'geometry' in element:
                coords = [(pt['lon'], pt['lat']) for pt in element['geometry']]
                if len(coords) >= 2:
                    street = StreetNetwork(
                        name=element.get('tags', {}).get('name', 'Unnamed'),
                        geometry=LineString(coords),
                        classification=element.get('tags', {}).get('highway', 'unknown')
                    )
                    streets.append(street)
        
        # Fallback to Census TIGER
        if not streets:
            tiger_data = self.open_data.query_census_tiger(bbox)
            for feature in tiger_data:
                if 'geometry' in feature and 'attributes' in feature:
                    geom = feature['geometry']
                    if geom['type'] == 'LineString':
                        coords = geom['coordinates']
                        street = StreetNetwork(
                            name=feature['attributes'].get('FULLNAME', 'Unnamed'),
                            geometry=LineString(coords),
                            classification=feature['attributes'].get('MTFCC', 'unknown')
                        )
                        streets.append(street)
        
        return streets
    
    def _pixels_to_coordinates(self, shape: np.ndarray, base_location: Dict, 
                              image_scale: float) -> List[Tuple[float, float]]:
        """Convert pixel coordinates to geographic coordinates"""
        
        # This is a simplified conversion - in practice, you'd need more sophisticated
        # georeferencing based on control points, scale, and projection
        
        base_lat = base_location.get('lat', 46.2)
        base_lon = base_location.get('lon', -122.7)
        
        # Estimate meters per pixel based on scale
        # At 1:2000 scale, 1 inch = 2000 inches on ground = ~167 feet = ~51 meters
        # Assuming 96 DPI, 1 pixel = 1/96 inch = ~0.53 meters at 1:2000
        meters_per_pixel = (image_scale / 96.0) * 0.0254  # Convert to meters
        
        # Convert to degrees (very rough approximation)
        lat_deg_per_meter = 1.0 / 111320.0  # meters per degree latitude
        lon_deg_per_meter = 1.0 / (111320.0 * np.cos(np.radians(base_lat)))
        
        coordinates = []
        for point in shape:
            x, y = point[0][0], point[0][1]  # OpenCV contour format
            
            # Convert pixels to offset in meters (assuming image center is base location)
            dx_meters = (x - 500) * meters_per_pixel  # Assume 1000px wide image
            dy_meters = (500 - y) * meters_per_pixel  # Flip Y axis
            
            # Convert to lat/lon
            new_lat = base_lat + (dy_meters * lat_deg_per_meter)
            new_lon = base_lon + (dx_meters * lon_deg_per_meter)
            
            coordinates.append((new_lon, new_lat))
        
        return coordinates
    
    def _calculate_area_sqft(self, polygon: Polygon) -> float:
        """Calculate area in square feet"""
        # Use UTM projection for accurate area calculation
        # This is simplified - should use appropriate UTM zone
        utm_proj = pyproj.Proj(proj='utm', zone=10, datum='WGS84')
        wgs84_proj = pyproj.Proj(proj='latlong', datum='WGS84')
        
        project = partial(pyproj.transform, wgs84_proj, utm_proj)
        utm_polygon = transform(project, polygon)
        
        area_sq_meters = utm_polygon.area
        area_sqft = area_sq_meters * 10.764  # Convert to square feet
        
        return area_sqft
    
    def _calculate_perimeter_ft(self, polygon: Polygon) -> float:
        """Calculate perimeter in feet"""
        # Similar to area calculation
        utm_proj = pyproj.Proj(proj='utm', zone=10, datum='WGS84')
        wgs84_proj = pyproj.Proj(proj='latlong', datum='WGS84')
        
        project = partial(pyproj.transform, wgs84_proj, utm_proj)
        utm_polygon = transform(project, polygon)
        
        perimeter_meters = utm_polygon.length
        perimeter_ft = perimeter_meters * 3.281  # Convert to feet
        
        return perimeter_ft
    
    def _calculate_street_match_confidence(self, polygon: Polygon, 
                                         streets: List[StreetNetwork]) -> float:
        """Calculate confidence based on street network alignment"""
        
        if not streets:
            return 0.5  # Default confidence when no street data
        
        boundary = polygon.boundary
        confidence_scores = []
        
        for street in streets:
            # Check if parcel boundary aligns with street
            distance = boundary.distance(street.geometry)
            
            # Streets should be close to parcel boundaries
            if distance < 0.0001:  # ~10 meters in degrees
                confidence_scores.append(0.9)
            elif distance < 0.0002:  # ~20 meters
                confidence_scores.append(0.7)
            else:
                confidence_scores.append(0.3)
        
        # Return average confidence, weighted by number of nearby streets
        if confidence_scores:
            return min(1.0, np.mean(confidence_scores) + 0.1 * len(confidence_scores))
        
        return 0.5
    
    def _parse_parcel_geometry(self, geometry: Dict) -> Polygon:
        """Parse parcel geometry from various formats"""
        
        if geometry['type'] == 'Polygon':
            coords = geometry['coordinates'][0]  # Exterior ring
            return Polygon(coords)
        elif geometry['type'] == 'MultiPolygon':
            # Use the largest polygon
            largest_area = 0
            largest_polygon = None
            for poly_coords in geometry['coordinates']:
                poly = Polygon(poly_coords[0])
                if poly.area > largest_area:
                    largest_area = poly.area
                    largest_polygon = poly
            return largest_polygon or Polygon()
        
        return Polygon()

class EnhancedParcelProcessor:
    """Main enhanced parcel processing class"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.open_data = OpenDataIntegrator()
        self.coord_processor = AdvancedCoordinateProcessor()
        self.shape_matcher = ShapeMatchingEngine(self.open_data, self.coord_processor)
    
    def process_parcel_map(self, image: np.ndarray, extracted_text: str, 
                          base_location: Dict) -> Dict[str, Any]:
        """Enhanced parcel map processing with open data integration"""
        
        self.logger.info("🚀 Starting enhanced parcel processing")
        
        # Extract location information
        location_info = self._extract_enhanced_location_info(extracted_text)
        
        # Detect coordinate system from any found coordinates
        detected_coords = location_info.get('coordinates', [])
        coord_system = self.coord_processor.detect_coordinate_system(detected_coords, base_location)
        
        # Enhanced shape detection
        shapes = self._detect_enhanced_shapes(image)
        
        # Estimate map scale
        image_scale = self._estimate_map_scale(extracted_text, image.shape)
        
        # Match shapes to street networks
        parcel_boundaries = self.shape_matcher.match_shapes_to_streets(
            shapes, base_location, image_scale
        )
        
        # Cross-reference with official parcel data
        if location_info.get('county') and location_info.get('state'):
            parcel_boundaries = self.shape_matcher.cross_reference_parcels(
                parcel_boundaries, 
                location_info['county'], 
                location_info['state']
            )
        
        # Generate comprehensive results
        results = {
            'enhanced_processing': True,
            'coordinate_system': {
                'detected': coord_system.name,
                'epsg_code': coord_system.epsg_code,
                'units': coord_system.unit
            },
            'parcel_boundaries': [
                {
                    'vertices': boundary.vertices,
                    'area_sqft': boundary.area_sqft,
                    'perimeter_ft': boundary.perimeter_ft,
                    'confidence': boundary.confidence,
                    'source': boundary.source
                }
                for boundary in parcel_boundaries
            ],
            'location_info': location_info,
            'map_scale': image_scale,
            'processing_quality': self._assess_processing_quality(parcel_boundaries, location_info),
            'recommendations': self._generate_recommendations(parcel_boundaries, location_info)
        }
        
        self.logger.info(f"✅ Enhanced processing complete. Found {len(parcel_boundaries)} parcels")
        
        return results
    
    def _extract_enhanced_location_info(self, text: str) -> Dict[str, Any]:
        """Enhanced location information extraction"""
        
        info = {
            'addresses': [],
            'streets': [],
            'coordinates': [],
            'county': None,
            'state': None,
            'lot_numbers': [],
            'parcel_ids': [],
            'subdivisions': []
        }
        
        # Enhanced regex patterns
        patterns = {
            'coordinates': [
                r'(\d+\.?\d*)[°\s]*(\d+\.?\d*)[\'′\s]*(\d+\.?\d*)[\"″\s]*([NSEW])',  # DMS
                r'(\d+\.\d+)[°\s]*([NSEW])',  # Decimal degrees
                r'(\d{6,7})\s*[,\s]\s*(\d{7,8})',  # UTM/State Plane
                r'(\d+\.\d{6,})[,\s]\s*(\d+\.\d{6,})',  # High precision lat/lon
            ],
            'streets': [
                r'(\d+\s+[A-Z][A-Za-z\s]+(?:STREET|ST|AVENUE|AVE|ROAD|RD|DRIVE|DR|LANE|LN|BOULEVARD|BLVD|CIRCLE|CIR|COURT|CT|WAY|PLACE|PL))',
                r'([A-Z][A-Za-z\s]+(?:STREET|ST|AVENUE|AVE|ROAD|RD|DRIVE|DR|LANE|LN|BOULEVARD|BLVD|CIRCLE|CIR|COURT|CT|WAY|PLACE|PL))',
            ],
            'county': r'([A-Z][A-Za-z\s]+)\s+COUNTY',
            'state': r'\b([A-Z]{2})\b|\b(WASHINGTON|OREGON|CALIFORNIA|IDAHO)\b',
            'lot_numbers': r'LOT\s+(\d+[A-Z]?)',
            'parcel_ids': r'PARCEL\s+(?:ID|#|NUMBER)[\s:]*([A-Z0-9\-]+)',
            'subdivisions': r'([A-Z][A-Za-z\s]+)\s+SUBDIVISION'
        }
        
        # Extract information using patterns
        for key, pattern_list in patterns.items():
            if isinstance(pattern_list, list):
                for pattern in pattern_list:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if key == 'coordinates':
                        info[key].extend(matches)
                    else:
                        info[key].extend([m if isinstance(m, str) else m[0] for m in matches])
            else:
                matches = re.findall(pattern_list, text, re.IGNORECASE)
                if matches:
                    if key in ['county', 'state']:
                        info[key] = matches[0] if isinstance(matches[0], str) else matches[0][0]
                    else:
                        info[key] = matches
        
        # Clean and deduplicate
        for key in ['streets', 'addresses', 'lot_numbers', 'parcel_ids', 'subdivisions']:
            info[key] = list(set(info[key]))  # Remove duplicates
        
        return info
    
    def _detect_enhanced_shapes(self, image: np.ndarray) -> List[np.ndarray]:
        """Enhanced shape detection with multiple algorithms"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        shapes = []
        
        # Method 1: Adaptive threshold + contours
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours1, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes.extend(contours1)
        
        # Method 2: Canny edge detection + contours
        edges = cv2.Canny(gray, 50, 150)
        contours2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes.extend(contours2)
        
        # Method 3: Morphological operations
        kernel = np.ones((3,3), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        contours3, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes.extend(contours3)
        
        # Filter and deduplicate shapes
        filtered_shapes = []
        for contour in shapes:
            area = cv2.contourArea(contour)
            if 1000 < area < 50000:  # Filter by reasonable parcel sizes
                # Approximate to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 4:  # Must be at least a quadrilateral
                    filtered_shapes.append(approx)
        
        # Remove duplicate shapes
        unique_shapes = self._remove_duplicate_shapes(filtered_shapes)
        
        return unique_shapes
    
    def _remove_duplicate_shapes(self, shapes: List[np.ndarray], tolerance: float = 20.0) -> List[np.ndarray]:
        """Remove duplicate shapes within tolerance"""
        
        unique_shapes = []
        
        for shape in shapes:
            is_duplicate = False
            
            for existing in unique_shapes:
                if self._shapes_similar(shape, existing, tolerance):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_shapes.append(shape)
        
        return unique_shapes
    
    def _shapes_similar(self, shape1: np.ndarray, shape2: np.ndarray, tolerance: float) -> bool:
        """Check if two shapes are similar within tolerance"""
        
        if len(shape1) != len(shape2):
            return False
        
        # Calculate centroids
        M1 = cv2.moments(shape1)
        M2 = cv2.moments(shape2)
        
        if M1['m00'] == 0 or M2['m00'] == 0:
            return False
        
        cx1, cy1 = M1['m10']/M1['m00'], M1['m01']/M1['m00']
        cx2, cy2 = M2['m10']/M2['m00'], M2['m01']/M2['m00']
        
        # Check if centroids are close
        distance = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
        
        return distance < tolerance
    
    def _estimate_map_scale(self, text: str, image_shape: Tuple[int, int]) -> float:
        """Estimate map scale from text and image"""
        
        # Look for explicit scale statements
        scale_patterns = [
            r'1:(\d+,?\d*)',
            r'SCALE\s*1:(\d+,?\d*)',
            r'(\d+,?\d*)\s*=\s*1\s*(?:INCH|IN)',
        ]
        
        for pattern in scale_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                scale_str = matches[0].replace(',', '')
                try:
                    return float(scale_str)
                except:
                    continue
        
        # Estimate from distance markers
        distance_pattern = r'(\d+\.?\d*)\s*(FT|FEET|MI|MILE|KM|METER|M)\b'
        distances = re.findall(distance_pattern, text, re.IGNORECASE)
        
        if distances:
            # Use image dimensions and found distances to estimate scale
            # This is a rough approximation
            return 2000  # Default reasonable scale for parcel maps
        
        return 2000  # Default scale
    
    def _assess_processing_quality(self, parcels: List[ParcelBoundary], location_info: Dict) -> Dict[str, Any]:
        """Assess the quality of processing results"""
        
        quality = {
            'overall_score': 0.0,
            'parcel_detection': 0.0,
            'location_accuracy': 0.0,
            'coordinate_precision': 0.0,
            'data_completeness': 0.0
        }
        
        # Parcel detection quality
        if parcels:
            avg_confidence = np.mean([p.confidence for p in parcels])
            quality['parcel_detection'] = avg_confidence
        
        # Location accuracy
        location_score = 0.0
        if location_info.get('county'):
            location_score += 0.3
        if location_info.get('state'):
            location_score += 0.3
        if location_info.get('streets'):
            location_score += 0.4
        quality['location_accuracy'] = location_score
        
        # Coordinate precision
        coord_score = 0.5  # Base score
        if location_info.get('coordinates'):
            coord_score = 0.8
        quality['coordinate_precision'] = coord_score
        
        # Data completeness
        completeness_score = 0.0
        total_fields = 6
        if parcels: completeness_score += 1
        if location_info.get('county'): completeness_score += 1
        if location_info.get('state'): completeness_score += 1
        if location_info.get('streets'): completeness_score += 1
        if location_info.get('coordinates'): completeness_score += 1
        if location_info.get('lot_numbers'): completeness_score += 1
        
        quality['data_completeness'] = completeness_score / total_fields
        
        # Overall score
        quality['overall_score'] = np.mean([
            quality['parcel_detection'],
            quality['location_accuracy'],
            quality['coordinate_precision'],
            quality['data_completeness']
        ])
        
        return quality
    
    def _generate_recommendations(self, parcels: List[ParcelBoundary], location_info: Dict) -> List[str]:
        """Generate recommendations for improving results"""
        
        recommendations = []
        
        if not parcels:
            recommendations.append("No parcels detected - try higher resolution image or better contrast")
        elif len(parcels) == 1:
            recommendations.append("Only one parcel detected - check if image contains multiple properties")
        
        if not location_info.get('county'):
            recommendations.append("County not detected - add county name to improve geocoding")
        
        if not location_info.get('coordinates'):
            recommendations.append("No coordinate references found - add coordinate grid or reference points")
        
        low_confidence_parcels = [p for p in parcels if p.confidence < 0.7]
        if low_confidence_parcels:
            recommendations.append(f"{len(low_confidence_parcels)} parcels have low confidence - verify boundaries manually")
        
        if not location_info.get('streets'):
            recommendations.append("No street names detected - add street labels for better validation")
        
        return recommendations


def create_enhanced_processor() -> EnhancedParcelProcessor:
    """Factory function to create enhanced processor"""
    return EnhancedParcelProcessor()

if __name__ == "__main__":
    # Test the enhanced processor
    processor = create_enhanced_processor()
    print("Enhanced Parcel Processor v2.0 initialized successfully!") 