#!/bin/bash
# start.sh - Script principal de inicio

set -e

echo "Iniciando AI Development Studio..."

# Verificar que Docker esté corriendo
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker no está corriendo"
    echo "Por favor inicia Docker Desktop y vuelve a intentar"
    exit 1
fi

# Verificar que Ollama esté corriendo
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Advertencia: Ollama no está corriendo en localhost:11434"
    echo "Asegúrate de que Ollama esté iniciado con tus modelos"
    echo "Continúando de todas formas..."
fi

# Crear directorios necesarios
mkdir -p data/postgres
mkdir -p data/redis
mkdir -p data/chroma
mkdir -p logs

# Copiar archivos de configuración si no existen
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Archivo .env creado desde .env.example"
fi

# Construir e iniciar servicios
echo "Construyendo servicios..."
docker-compose build --parallel

echo "Iniciando servicios..."
docker-compose up -d

# Esperar a que los servicios estén listos
echo "Esperando a que los servicios estén listos..."

# Esperar PostgreSQL
echo "Esperando PostgreSQL..."
until docker-compose exec -T postgres pg_isready -U user -d ai_dev_studio; do
    sleep 2
done

# Esperar Redis
echo "Esperando Redis..."
until docker-compose exec -T redis redis-cli ping; do
    sleep 2
done

# Esperar servicios backend
echo "Esperando servicios backend..."
until curl -s http://localhost:8000/health > /dev/null; do
    sleep 2
done

echo "AI Development Studio está listo!"
echo ""
echo "Frontend: http://localhost:3000"
echo "API Gateway: http://localhost:8000"
echo "Traefik Dashboard: http://localhost:8080"
echo "PostgreSQL: localhost:5433"
echo "Redis: localhost:6380"
echo ""
echo "Para ver logs: docker-compose logs -f"
echo "Para detener: ./scripts/stop.sh"
