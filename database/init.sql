-- AI Development Studio Database Schema
-- Inicialización de PostgreSQL

-- Crear extensiones necesarias
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Tabla de proyectos
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    architecture_type VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    config JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active'
);

-- Tabla de estados de sesión
CREATE TABLE IF NOT EXISTS session_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    session_start TIMESTAMP DEFAULT NOW(),
    session_end TIMESTAMP,
    open_files JSONB DEFAULT '[]',
    cursor_positions JSONB DEFAULT '{}',
    current_task TEXT,
    ai_conversations JSONB DEFAULT '[]',
    model_preferences JSONB DEFAULT '{}',
    session_data JSONB DEFAULT '{}'
);

-- Tabla de historial de conversaciones
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    session_id UUID REFERENCES session_states(id) ON DELETE SET NULL,
    message_type VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    model_used VARCHAR(100),
    sources_used JSONB DEFAULT '[]',
    timestamp TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Tabla de performance de modelos
CREATE TABLE IF NOT EXISTS model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    task_type VARCHAR(100),
    response_time_ms INTEGER,
    memory_usage_mb INTEGER,
    success_rate DECIMAL(3,2),
    user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Tabla de documentos en knowledge base
CREATE TABLE IF NOT EXISTS knowledge_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    file_size INTEGER,
    upload_date TIMESTAMP DEFAULT NOW(),
    processing_status VARCHAR(50) DEFAULT 'pending',
    chunk_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'
);

-- Índices para optimizar consultas
CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);
CREATE INDEX IF NOT EXISTS idx_projects_updated_at ON projects(updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_session_states_project_id ON session_states(project_id);
CREATE INDEX IF NOT EXISTS idx_session_states_start ON session_states(session_start DESC);

CREATE INDEX IF NOT EXISTS idx_conversations_project_id ON conversations(project_id);
CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_type ON conversations(message_type);

CREATE INDEX IF NOT EXISTS idx_model_performance_name ON model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp ON model_performance(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_knowledge_docs_project_id ON knowledge_documents(project_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_docs_status ON knowledge_documents(processing_status);

-- Trigger para actualizar updated_at automáticamente
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_projects_updated_at 
    BEFORE UPDATE ON projects 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Datos de ejemplo para desarrollo
INSERT INTO projects (name, description, architecture_type, config) VALUES 
    ('AI Chat Agent', 'Agente conversacional con LangChain', 'microservices', 
     '{"models": ["deepseek-coder:33b"], "frameworks": ["langchain"]}'),
    ('ML Pipeline', 'Pipeline de Machine Learning automatizado', 'data-pipeline',
     '{"models": ["mixtral:8x22b"], "tools": ["pytorch", "scikit-learn"]}'),
    ('Code Analyzer', 'Analizador estático de código Python', 'analysis-tool',
     '{"models": ["deepseek-coder:33b"], "languages": ["python"]}')
ON CONFLICT DO NOTHING;

-- Mostrar resumen de inicialización
DO $$
BEGIN
    RAISE NOTICE 'Base de datos AI Dev Studio inicializada correctamente';
    RAISE NOTICE 'Tablas creadas: projects, session_states, conversations, model_performance, knowledge_documents';
    RAISE NOTICE 'Sistema listo para usar';
END $$;
