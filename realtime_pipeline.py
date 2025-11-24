"""
=================================================================================
REAL-TIME Feature Extraction API - PRODUCTION READY
=================================================================================
✅ FastAPI with proper localhost configuration
✅ Real-time soil data with NPK values
✅ Intelligent buffer strategy
✅ Ready for model integration
✅ ML Engineers endpoint added
=================================================================================
"""

import ee
from datetime import datetime
import math
import requests
import logging
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# ==================== CONFIGURATION ====================

class Config:
    GEE_PROJECT_ID = 'grad-project-470219'
    OPENWEATHER_API_KEY = "96e35595ffc622fdda9a5639c39c22b7"

# ==================== FASTAPI MODELS ====================

class CoordinateRequest(BaseModel):
    latitude: float
    longitude: float

class FeatureResponse(BaseModel):
    success: bool
    latitude: float
    longitude: float
    location_name: str
    query_timestamp: str
    data_quality_score: float
    
    # Soil Properties
    sand: Optional[float]  # g/kg
    clay: Optional[float]  # g/kg
    soc: Optional[float]   # cg/kg
    ph: Optional[float]    # pH * 10
    cec: Optional[float]   # mmol(c)/kg
    
    # NPK Nutrients
    nitrogen_percent: Optional[float]
    phosphorus_mg_kg: Optional[float]
    potassium_mg_kg: Optional[float]
    npk_confidence: Optional[str]
    
    # Vegetation
    ndvi: Optional[float]
    ndvi_date: Optional[str]
    
    # Climate
    temperature_c: Optional[float]
    relative_humidity_percent: Optional[float]
    precipitation_m: Optional[float]
    solar_radiation_j_m2: Optional[float]
    
    # Metadata
    soil_retrieval_method: Optional[str]
    soil_buffer_used_m: Optional[int]
    weather_source: Optional[str]

# ==================== ML ENGINEERS MODELS ====================

class MLFeaturesRequest(BaseModel):
    latitude: float
    longitude: float

class MLFeaturesResponse(BaseModel):
    success: bool
    features: Dict
    metadata: Dict

# ==================== REAL-TIME EXTRACTOR ====================

class RealTimeFeatureExtractor:
    def __init__(self):
        self.logger = self._setup_logging()
        self._initialize_ee()
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
        return logging.getLogger(__class__.__name__)
    
    def _initialize_ee(self):
        try:
            ee.Initialize(project=Config.GEE_PROJECT_ID)
            self.logger.info("✅ Earth Engine initialized")
        except Exception as e:
            self.logger.error(f"❌ Earth Engine failed: {e}")
            raise
    
    def compute_dewpoint(self, t_celsius, rh_percent):
        try:
            if t_celsius is None or rh_percent is None: return None
            a, b = 17.27, 237.7
            alpha = ((a * t_celsius) / (b + t_celsius)) + math.log(rh_percent / 100.0)
            return (b * alpha) / (a - alpha)
        except: return None

    def _estimate_npk_nutrients(self, soil_props: Dict) -> Dict:
        try:
            soc = soil_props.get('soc', 0) or 0
            cec = soil_props.get('cec', 0) or 0
            clay = soil_props.get('clay', 0) or 0
            silt = soil_props.get('silt', 0) or 0
            ph = soil_props.get('ph', 70) or 70
            
            soc = float(soc) if soc else 0
            cec = float(cec) if cec else 0
            clay = float(clay) if clay else 0
            silt = float(silt) if silt else 0
            ph_actual = ph / 10.0
            
            # Nitrogen
            if soc > 0:
                soc_percent = soc / 100.0
                nitrogen_percent = soc_percent / 11.0
                nitrogen_percent = max(0.05, min(0.30, nitrogen_percent))
            else:
                nitrogen_percent = 0.12
            
            # Phosphorus
            base_p = (soc * 0.25) if soc > 0 else 15.0
            cec_factor = 1.0 + ((cec - 200) / 1000.0) if cec else 1.0
            clay_factor = 1.0 - ((clay - 250) / 2000.0) if clay else 1.0
            ph_factor = 1.2 if (6.0 <= ph_actual <= 7.5) else (0.9 if ph_actual > 7.5 else 0.95)
            phosphorus_mg_kg = base_p * cec_factor * clay_factor * ph_factor
            phosphorus_mg_kg = max(5.0, min(50.0, phosphorus_mg_kg))
            
            # Potassium
            base_k = (cec * 0.8) if cec > 0 else 150.0
            clay_factor_k = 1.0 + ((clay - 250) / 500.0) if clay else 1.0
            silt_factor = 1.0 + ((silt - 300) / 1000.0) if silt else 1.0
            potassium_mg_kg = base_k * clay_factor_k * silt_factor
            potassium_mg_kg = max(50.0, min(400.0, potassium_mg_kg))
            
            return {
                'nitrogen_percent': round(nitrogen_percent, 3),
                'phosphorus_mg_kg': round(phosphorus_mg_kg, 1),
                'potassium_mg_kg': round(potassium_mg_kg, 1),
                'npk_confidence': 'Medium' if soc > 0 and cec > 0 else 'Low'
            }
        except:
            return {
                'nitrogen_percent': 0.12, 'phosphorus_mg_kg': 18.5, 'potassium_mg_kg': 180.0,
                'npk_confidence': 'Low'
            }

    def get_realtime_weather(self, latitude, longitude):
        if not Config.OPENWEATHER_API_KEY: 
            self.logger.warning("⚠️ OpenWeather API key not set")
            return None
        
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': latitude, 
                'lon': longitude, 
                'appid': Config.OPENWEATHER_API_KEY, 
                'units': 'metric'
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            dewpoint = self.compute_dewpoint(temp, humidity)
            
            dt = datetime.fromtimestamp(data['dt'])
            sunrise = datetime.fromtimestamp(data['sys']['sunrise'])
            sunset = datetime.fromtimestamp(data['sys']['sunset'])
            hour = dt.hour
            is_daytime = sunrise.hour <= hour <= sunset.hour
            clouds = data['clouds']['all']
            
            if is_daytime:
                cloud_factor = (100 - clouds) / 100
                time_factor = 1 - abs(hour - 12) / 6
                solar_radiation_wm2 = 1000 * cloud_factor * max(time_factor, 0.1)
            else:
                solar_radiation_wm2 = 0
            
            return {
                'timestamp': dt.isoformat(),
                't2m_c': temp, 
                'td2m_c': dewpoint, 
                'rh_pct': humidity,
                'tp_m': data.get('rain', {}).get('1h', 0) / 1000,
                'ssrd_jm2': solar_radiation_wm2 * 3600,
                'location_name': data.get('name', 'Unknown'),
                'country': data['sys'].get('country', 'Unknown'),
                'weather_description': data['weather'][0]['description'],
                'pressure_hpa': data['main']['pressure'],
                'wind_speed_ms': data['wind']['speed']
            }
        except Exception as e:
            self.logger.error(f"❌ Weather data failed: {e}")
            return None

    def get_soil_data_with_buffer(self, latitude, longitude):
        try:
            point = ee.Geometry.Point([longitude, latitude])
            buffer_sizes = [0, 500, 1000, 2000]
            
            # Load soil datasets
            sand = ee.Image("projects/soilgrids-isric/sand_mean").select("sand_0-5cm_mean")
            silt = ee.Image("projects/soilgrids-isric/silt_mean").select("silt_0-5cm_mean")
            clay = ee.Image("projects/soilgrids-isric/clay_mean").select("clay_0-5cm_mean")
            soc = ee.Image("projects/soilgrids-isric/soc_mean").select("soc_0-5cm_mean")
            ph = ee.Image("projects/soilgrids-isric/phh2o_mean").select("phh2o_0-5cm_mean")
            cec = ee.Image("projects/soilgrids-isric/cec_mean").select("cec_0-5cm_mean")
            
            soil_image = ee.Image.cat([sand, silt, clay, soc, ph, cec]) \
                .rename(['sand', 'silt', 'clay', 'soc', 'ph', 'cec'])
            
            for buffer_size in buffer_sizes:
                try:
                    geometry = point if buffer_size == 0 else point.buffer(buffer_size)
                    method = "exact_point" if buffer_size == 0 else f"buffer_{buffer_size}m"
                    
                    soil_data = soil_image.reduceRegion(
                        reducer=ee.Reducer.mean(), 
                        geometry=geometry, 
                        scale=250, 
                        bestEffort=True
                    )
                    
                    soil_values = {
                        'sand': soil_data.get('sand').getInfo(),
                        'silt': soil_data.get('silt').getInfo(),
                        'clay': soil_data.get('clay').getInfo(),
                        'soc': soil_data.get('soc').getInfo(),
                        'ph': soil_data.get('ph').getInfo(),
                        'cec': soil_data.get('cec').getInfo()
                    }
                    
                    if soil_values['sand'] is not None:
                        self.logger.info(f"✅ Soil data success with {method}")
                        npk_data = self._estimate_npk_nutrients(soil_values)
                        soil_values.update(npk_data)
                        soil_values['soil_retrieval_method'] = method
                        soil_values['soil_buffer_used_m'] = buffer_size
                        return soil_values
                except Exception as e:
                    self.logger.warning(f"⚠️ Soil buffer {buffer_size}m failed: {str(e)[:50]}")
                    continue
            
            self.logger.warning("❌ All soil retrieval methods failed")
            return {
                'sand': None, 'silt': None, 'clay': None, 'soc': None, 'ph': None, 'cec': None,
                'nitrogen_percent': 0.12, 'phosphorus_mg_kg': 18.5, 'potassium_mg_kg': 180.0,
                'npk_confidence': 'Low - Default', 'soil_retrieval_method': 'no_data_available'
            }
        except Exception as e:
            self.logger.error(f"❌ Soil extraction failed: {e}")
            return None

    def get_ndvi_data(self, latitude, longitude):
        try:
            point = ee.Geometry.Point([longitude, latitude])
            end_date = ee.Date(datetime.now())
            start_date = end_date.advance(-60, 'day')
            
            ndvi_col = ee.ImageCollection('MODIS/061/MOD13A2').select('NDVI') \
                .filterDate(start_date, end_date).filterBounds(point)
            
            if ndvi_col.size().getInfo() > 0:
                ndvi_latest = ndvi_col.sort('system:time_start', False).first()
                ndvi_value = ndvi_latest.multiply(0.0001).reduceRegion(
                    reducer=ee.Reducer.first(), geometry=point, scale=1000
                ).get('NDVI').getInfo()
                
                if ndvi_value is not None:
                    ndvi_date = ee.Date(ndvi_latest.get('system:time_start')).format('YYYY-MM-dd').getInfo()
                    self.logger.info(f"✅ NDVI: {ndvi_value:.3f}")
                    return {'ndvi': ndvi_value, 'ndvi_date': ndvi_date}
            
            self.logger.warning("⚠️ No NDVI data available")
            return {'ndvi': None, 'ndvi_date': None}
        except Exception as e:
            self.logger.error(f"❌ NDVI failed: {e}")
            return {'ndvi': None, 'ndvi_date': None}

    def extract_features(self, latitude: float, longitude: float) -> Dict:
        self.logger.info(f"🚀 Extracting features for ({latitude:.4f}, {longitude:.4f})")
        
        query_time = datetime.now()
        
        # Get all data sources
        weather_data = self.get_realtime_weather(latitude, longitude)
        soil_data = self.get_soil_data_with_buffer(latitude, longitude)
        ndvi_data = self.get_ndvi_data(latitude, longitude)
        
        # Calculate data quality
        data_quality_score = self._calculate_data_quality(soil_data, ndvi_data, weather_data)
        
        features = {
            'success': True, 
            'latitude': latitude, 
            'longitude': longitude,
            'location_name': weather_data['location_name'] if weather_data else 'Unknown',
            'query_timestamp': query_time.isoformat(),
            'data_quality_score': data_quality_score,
            
            # Soil Properties
            'sand': soil_data.get('sand'), 
            'clay': soil_data.get('clay'),
            'soc': soil_data.get('soc'), 
            'ph': soil_data.get('ph'), 
            'cec': soil_data.get('cec'),
            
            # NPK Nutrients
            'nitrogen_percent': soil_data.get('nitrogen_percent'),
            'phosphorus_mg_kg': soil_data.get('phosphorus_mg_kg'),
            'potassium_mg_kg': soil_data.get('potassium_mg_kg'),
            'npk_confidence': soil_data.get('npk_confidence'),
            
            # Vegetation
            'ndvi': ndvi_data.get('ndvi'), 
            'ndvi_date': ndvi_data.get('ndvi_date'),
            
            # Climate
            'temperature_c': weather_data.get('t2m_c') if weather_data else None,
            'relative_humidity_percent': weather_data.get('rh_pct') if weather_data else None,
            'precipitation_m': weather_data.get('tp_m') if weather_data else None,
            'solar_radiation_j_m2': weather_data.get('ssrd_jm2') if weather_data else None,
            
            # Metadata
            'soil_retrieval_method': soil_data.get('soil_retrieval_method'),
            'soil_buffer_used_m': soil_data.get('soil_buffer_used_m'),
            'weather_source': 'OpenWeatherMap' if weather_data else None
        }
        
        self.logger.info("✅ Feature extraction completed")
        return features

    def _calculate_data_quality(self, soil_data, ndvi_data, weather_data) -> float:
        scores = [
            1.0 if soil_data and soil_data.get('sand') is not None else 0.0,
            1.0 if ndvi_data.get('ndvi') is not None else 0.0,
            1.0 if weather_data and weather_data.get('t2m_c') is not None else 0.0
        ]
        return (sum(scores) / len(scores)) * 100

# ==================== FASTAPI APP ====================

extractor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global extractor
    extractor = RealTimeFeatureExtractor()
    yield
    # Shutdown - cleanup if needed

app = FastAPI(
    title="Real-Time Feature Extraction API", 
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {
        "message": "Real-Time Feature Extraction API", 
        "status": "running",
        "endpoints": {
            "health": "/health",
            "extract_features": "/extract-features",
            "ml_features": "/extract-ml-features"
        }
    }

@app.post("/extract-features", response_model=FeatureResponse)
async def extract_features(request: CoordinateRequest):
    """Extract real-time features for coordinates"""
    try:
        if not extractor:
            raise HTTPException(status_code=500, detail="Service not initialized")
        
        features = extractor.extract_features(
            latitude=request.latitude, 
            longitude=request.longitude
        )
        return FeatureResponse(**features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-ml-features", response_model=MLFeaturesResponse)
async def extract_ml_features(request: MLFeaturesRequest):
    """
    🎯 YOUR MAIN ENDPOINT for ML Engineers
    This is what ML engineers will call to get features for their models
    """
    try:
        if not extractor:
            raise HTTPException(status_code=500, detail="Service not initialized")
        
        # Use your existing extract_features function
        features = extractor.extract_features(
            latitude=request.latitude,
            longitude=request.longitude
        )
        
        # Format specifically for ML models
        ml_ready_features = {
            # Soil Features
            'sand': features.get('sand'),
            'clay': features.get('clay'), 
            'soc': features.get('soc'),
            'ph': features.get('ph'),
            'cec': features.get('cec'),
            
            # NPK Nutrients
            'nitrogen': features.get('nitrogen_percent'),
            'phosphorus': features.get('phosphorus_mg_kg'),
            'potassium': features.get('potassium_mg_kg'),
            
            # Vegetation
            'ndvi': features.get('ndvi'),
            
            # Climate
            'temperature': features.get('temperature_c'),
            'humidity': features.get('relative_humidity_percent'),
            'precipitation': features.get('precipitation_m'),
            'solar_radiation': features.get('solar_radiation_j_m2'),
            
            # Location
            'latitude': features.get('latitude'),
            'longitude': features.get('longitude')
        }
        
        return MLFeaturesResponse(
            success=True,
            features=ml_ready_features,
            metadata={
                'location_name': features.get('location_name'),
                'query_timestamp': features.get('query_timestamp'),
                'data_quality': features.get('data_quality_score'),
                'soil_retrieval_method': features.get('soil_retrieval_method'),
                'npk_confidence': features.get('npk_confidence')
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "service": "feature-extraction"
    }

if __name__ == "__main__":
    print("🚀 Starting Real-Time Feature Extraction API...")
    print("📝 Access the API at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🎯 ML Engineers Endpoint: POST /extract-ml-features")
    print("⏹️  Press CTRL+C to stop the server")
    
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Allows external connections
        port=8000,
        reload=True
    )