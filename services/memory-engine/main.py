#!/usr/bin/env python3
"""
AI Development Studio - Memory Engine Service
Sistema de memoria y persistencia de sesiones
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/ai_dev_studio")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

class Project(BaseModel):
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    architecture_type: Optional[str] = None
    config: Optional[Dict] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class SessionState(BaseModel):
    id: Optional[str] = None
    project_id: str
    open_files: Optional[List[Dict]] = None
    cursor_positions: Optional[Dict] = None
    current_task: Optional[str] = None
    ai_conversations: Optional[List[Dict]] = None
    model_preferences: Optional[Dict] = None
    session_data: Optional[Dict] = None

class MemoryEngine:
    """Motor de memoria y persistencia"""
    
    def __init__(self):
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        
    async def initialize(self):
        """Inicializa conexiones a base de datos"""
        try:
            self.db_pool = await asyncpg.create_pool(DATABASE_URL)
            logger.info("Conectado a PostgreSQL")
            
            self.redis_client = redis.from_url(REDIS_URL)
            await self.redis_client.ping()
            logger.info("Conectado a Redis")
            
            logger.info("Memory Engine inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando Memory Engine: {e}")
            raise
async def create_project(self, project: Project) -> Project:
        """Crea un nuevo proyecto"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    INSERT INTO projects (name, description, architecture_type, config)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id, created_at, updated_at
                """
                result = await conn.fetchrow(
                    query,
                    project.name,
                    project.description,
                    project.architecture_type,
                    json.dumps(project.config or {})
                )
                
                project.id = str(result['id'])
                project.created_at = result['created_at']
                project.updated_at = result['updated_at']
                
                logger.info(f"Proyecto creado: {project.name} ({project.id})")
                return project
                
        except Exception as e:
            logger.error(f"Error creando proyecto: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_project(self, project_id: str) -> Optional[Project]:
        """Obtiene un proyecto por ID"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT id, name, description, architecture_type, config, 
                           created_at, updated_at
                    FROM projects 
                    WHERE id = $1
                """
                result = await conn.fetchrow(query, project_id)
                
                if result:
                    return Project(
                        id=str(result['id']),
                        name=result['name'],
                        description=result['description'],
                        architecture_type=result['architecture_type'],
                        config=json.loads(result['config'] or '{}'),
                        created_at=result['created_at'],
                        updated_at=result['updated_at']
                    )
                return None
                
        except Exception as e:
            logger.error(f"Error obteniendo proyecto {project_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def list_projects(self) -> List[Project]:
        """Lista todos los proyectos"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT id, name, description, architecture_type, config,
                           created_at, updated_at
                    FROM projects 
                    ORDER BY updated_at DESC
                """
                results = await conn.fetch(query)
                
                projects = []
                for result in results:
                    projects.append(Project(
                        id=str(result['id']),
                        name=result['name'],
                        description=result['description'],
                        architecture_type=result['architecture_type'],
                        config=json.loads(result['config'] or '{}'),
                        created_at=result['created_at'],
                        updated_at=result['updated_at']
                    ))
                
                return projects
                
        except Exception as e:
            logger.error(f"Error listando proyectos: {e}")
            raise HTTPException(status_code=500, detail=str(e))
async def save_session_state(self, session: SessionState) -> SessionState:
        """Guarda estado de sesión"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    INSERT INTO session_states 
                    (project_id, open_files, cursor_positions, current_task, 
                     ai_conversations, model_preferences, session_data)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id, session_start
                """
                result = await conn.fetchrow(
                    query,
                    session.project_id,
                    json.dumps(session.open_files or []),
                    json.dumps(session.cursor_positions or {}),
                    session.current_task,
                    json.dumps(session.ai_conversations or []),
                    json.dumps(session.model_preferences or {}),
                    json.dumps(session.session_data or {})
                )
                
                session.id = str(result['id'])
                
                # Cache en Redis para acceso rápido
                if self.redis_client:
                    cache_key = f"session:{session.project_id}:current"
                    await self.redis_client.setex(
                        cache_key, 
                        3600,  # 1 hora
                        json.dumps(session.dict())
                    )
                
                logger.info(f"Sesión guardada: {session.id}")
                return session
                
        except Exception as e:
            logger.error(f"Error guardando sesión: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def restore_session(self, project_id: str) -> Optional[SessionState]:
        """Restaura último estado de sesión para un proyecto"""
        try:
            # Revisar cache de Redis primero
            if self.redis_client:
                cache_key = f"session:{project_id}:current"
                cached_session = await self.redis_client.get(cache_key)
                if cached_session:
                    data = json.loads(cached_session)
                    logger.info(f"Sesión restaurada desde cache: {project_id}")
                    return SessionState(**data)
            
            # Si no está en cache, buscar en DB
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT id, project_id, open_files, cursor_positions, current_task,
                           ai_conversations, model_preferences, session_data, session_start
                    FROM session_states 
                    WHERE project_id = $1
                    ORDER BY session_start DESC
                    LIMIT 1
                """
                result = await conn.fetchrow(query, project_id)
                
                if result:
                    session = SessionState(
                        id=str(result['id']),
                        project_id=result['project_id'],
                        open_files=json.loads(result['open_files'] or '[]'),
                        cursor_positions=json.loads(result['cursor_positions'] or '{}'),
                        current_task=result['current_task'],
                        ai_conversations=json.loads(result['ai_conversations'] or '[]'),
                        model_preferences=json.loads(result['model_preferences'] or '{}'),
                        session_data=json.loads(result['session_data'] or '{}')
                    )
                    
                    logger.info(f"Sesión restaurada desde DB: {project_id}")
                    return session
                
                return None
                
        except Exception as e:
            logger.error(f"Error restaurando sesión {project_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Instancia global del motor
memory_engine = MemoryEngine()

# Lifecycle events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await memory_engine.initialize()
    yield
    # Shutdown
    if memory_engine.db_pool:
        await memory_engine.db_pool.close()
    if memory_engine.redis_client:
        await memory_engine.redis_client.close()
# FastAPI app
app = FastAPI(
    title="AI Dev Studio - Memory Engine",
    version="1.0.0",
    description="Sistema de memoria y persistencia",
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
        "service": "memory-engine",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/memory/projects", response_model=Project)
async def create_project(project: Project):
    """Crea un nuevo proyecto"""
    return await memory_engine.create_project(project)

@app.get("/api/v1/memory/projects", response_model=List[Project])
async def list_projects():
    """Lista todos los proyectos"""
    return await memory_engine.list_projects()

@app.get("/api/v1/memory/projects/{project_id}", response_model=Project)
async def get_project(project_id: str):
    """Obtiene un proyecto específico"""
    project = await memory_engine.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")
    return project

@app.post("/api/v1/memory/sessions", response_model=SessionState)
async def save_session(session: SessionState):
    """Guarda estado de sesión"""
    return await memory_engine.save_session_state(session)

@app.get("/api/v1/memory/sessions/{project_id}", response_model=SessionState)
async def restore_session(project_id: str):
    """Restaura último estado de sesión"""
    session = await memory_engine.restore_session(project_id)
    if not session:
        raise HTTPException(status_code=404, detail="No hay sesiones guardadas")
    return session

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
