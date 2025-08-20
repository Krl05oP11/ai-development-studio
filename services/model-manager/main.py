#!/usr/bin/env python3
"""
AI Development Studio - Model Manager Service
Gestión inteligente de modelos Ollama con auto-discovery y smart routing
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import aiohttp
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configuración de logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuración
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Modelos de datos
class ModelInfo(BaseModel):
    name: str
    size: str
    modified: str
    digest: str
    details: Optional[Dict] = None

class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = None
    project_id: Optional[str] = None
    context: Optional[Dict] = None

class QueryResponse(BaseModel):
    response: str
    model_used: str
    response_time_ms: int
    sources_used: Optional[List[str]] = None
# Configuración de especialización de modelos
MODEL_SPECIALTIES = {
    "deepseek-coder:33b": {
        "specialty": "Python, ML, PyTorch, debugging, optimization",
        "use_cases": ["coding", "debugging", "optimization", "python", "ml"],
        "context_window": 8192,
        "priority": 1
    },
    "mixtral:8x22b": {
        "specialty": "System design, patterns, scalability, microservices",
        "use_cases": ["architecture", "design", "reasoning", "planning"],
        "context_window": 32768,
        "priority": 2
    },
    "gpt-oss:20b": {
        "specialty": "Error analysis, quick fixes, troubleshooting",
        "use_cases": ["debug", "quick", "fixes", "troubleshooting"],
        "context_window": 4096,
        "priority": 3
    },
    "qwen2.5:14b": {
        "specialty": "General reasoning and analysis",
        "use_cases": ["general", "reasoning", "analysis"],
        "context_window": 8192,
        "priority": 4
    },
    "llava:7b": {
        "specialty": "Image analysis and vision tasks",
        "use_cases": ["image", "vision", "multimodal"],
        "context_window": 4096,
        "priority": 5
    }
}

class ModelManager:
    """Gestor principal de modelos Ollama"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.available_models: Dict[str, ModelInfo] = {}
        self.discovery_running = False
        
    async def initialize(self):
        """Inicializa el gestor de modelos"""
        try:
            # Conectar a Redis
            self.redis_client = redis.from_url(REDIS_URL)
            await self.redis_client.ping()
            logger.info("✅ Conectado a Redis")
            
            # Descubrir modelos iniciales
            await self.discover_models()
            
            # Iniciar discovery en background
            if not self.discovery_running:
                asyncio.create_task(self.auto_discovery_loop())
                
            logger.info("🤖 Model Manager inicializado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Model Manager: {e}")
            raise
    
    async def discover_models(self) -> List[ModelInfo]:
        """Descubre modelos disponibles en Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{OLLAMA_BASE_URL}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = []
                        
                        for model_data in data.get("models", []):
                            model = ModelInfo(
                                name=model_data["name"],
                                size=model_data.get("size", "unknown"),
                                modified=model_data.get("modified_at", ""),
                                digest=model_data.get("digest", ""),
                                details=model_data.get("details", {})
                            )
                            models.append(model)
                            self.available_models[model.name] = model
                        
                        # Cachear en Redis
                        if self.redis_client:
                            await self.redis_client.setex(
                                "available_models",
                                300,  # 5 minutos
                                json.dumps([model.dict() for model in models])
                            )
                        
                        logger.info(f"🔍 Descubiertos {len(models)} modelos: {[m.name for m in models]}")
                        return models
                    else:
                        logger.error(f"❌ Error conectando con Ollama: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"❌ Error descubriendo modelos: {e}")
            return []
async def auto_discovery_loop(self):
        """Loop de auto-discovery que corre en background"""
        self.discovery_running = True
        logger.info("🔄 Iniciando auto-discovery de modelos (cada 5 minutos)")
        
        while self.discovery_running:
            try:
                await asyncio.sleep(300)  # 5 minutos
                await self.discover_models()
            except Exception as e:
                logger.error(f"❌ Error en auto-discovery: {e}")
                await asyncio.sleep(60)  # Retry en 1 minuto si hay error
    
    def smart_model_selection(self, query: str, context: Optional[Dict] = None) -> str:
        """Selección inteligente de modelo basada en el query y contexto"""
        query_lower = query.lower()
        
        # Buscar keywords específicas
        for model_name, config in MODEL_SPECIALTIES.items():
            if model_name in self.available_models:
                for use_case in config["use_cases"]:
                    if use_case in query_lower:
                        logger.info(f"🧠 Modelo seleccionado: {model_name} (keyword: {use_case})")
                        return model_name
        
        # Si no hay match específico, usar deepseek-coder por defecto (mejor modelo)
        default_model = "deepseek-coder:33b"
        if default_model in self.available_models:
            logger.info(f"🧠 Modelo por defecto: {default_model}")
            return default_model
        
        # Fallback al primer modelo disponible
        if self.available_models:
            fallback = list(self.available_models.keys())[0]
            logger.info(f"🧠 Modelo fallback: {fallback}")
            return fallback
        
        raise HTTPException(status_code=503, detail="No hay modelos disponibles")
    
    async def query_model(self, query: str, model_name: str, context: Optional[Dict] = None) -> QueryResponse:
        """Envía query a un modelo específico"""
        start_time = datetime.now()
        
        try:
            payload = {
                "model": model_name,
                "prompt": query,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        end_time = datetime.now()
                        response_time = int((end_time - start_time).total_seconds() * 1000)
                        
                        return QueryResponse(
                            response=data.get("response", ""),
                            model_used=model_name,
                            response_time_ms=response_time,
                            sources_used=[]
                        )
                    else:
                        logger.error(f"❌ Error en modelo {model_name}: {response.status}")
                        raise HTTPException(status_code=response.status, detail="Error en el modelo")
                        
        except Exception as e:
            logger.error(f"❌ Error consultando modelo {model_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Instancia global del gestor
model_manager = ModelManager()

# Lifecycle events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await model_manager.initialize()
    yield
    # Shutdown
    model_manager.discovery_running = False
    if model_manager.redis_client:
        await model_manager.redis_client.close()
# FastAPI app
app = FastAPI(
    title="AI Dev Studio - Model Manager",
    version="1.0.0",
    description="Gestión inteligente de modelos Ollama",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "model-manager",
        "timestamp": datetime.now().isoformat(),
        "models_available": len(model_manager.available_models)
    }

@app.get("/api/v1/models", response_model=List[ModelInfo])
async def list_models():
    """Lista todos los modelos disponibles"""
    models = list(model_manager.available_models.values())
    return models

@app.get("/api/v1/models/{model_name}/info")
async def get_model_info(model_name: str):
    """Obtiene información detallada de un modelo"""
    if model_name not in model_manager.available_models:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    model_info = model_manager.available_models[model_name]
    specialty_info = MODEL_SPECIALTIES.get(model_name, {})
    
    return {
        "model": model_info,
        "specialty": specialty_info
    }

@app.post("/api/v1/models/query", response_model=QueryResponse)
async def query_models(request: QueryRequest):
    """Envía query con selección automática o manual de modelo"""
    
    # Selección de modelo
    if request.model:
        model_name = request.model
        if model_name not in model_manager.available_models:
            raise HTTPException(status_code=404, detail=f"Modelo {model_name} no disponible")
    else:
        model_name = model_manager.smart_model_selection(request.query, request.context)
    
    # Ejecutar query
    response = await model_manager.query_model(request.query, model_name, request.context)
    return response

@app.post("/api/v1/models/discover")
async def force_discovery(background_tasks: BackgroundTasks):
    """Fuerza redescubrimiento de modelos"""
    background_tasks.add_task(model_manager.discover_models)
    return {"message": "Discovery iniciado en background"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
