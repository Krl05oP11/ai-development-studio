#!/bin/bash
# logs.sh - Ver logs de servicios

if [ $# -eq 0 ]; then
    echo "Mostrando logs de todos los servicios..."
    docker-compose logs -f
else
    echo "Mostrando logs de $1..."
    docker-compose logs -f $1
fi
