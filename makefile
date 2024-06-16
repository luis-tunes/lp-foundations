# Makefile

# Variables
DOCKER_COMPOSE = docker-compose

# Default target
all: up

# Build Docker containers
build:
    $(DOCKER_COMPOSE) build

# Run Docker containers in detached mode
up:
    $(DOCKER_COMPOSE) up --build -d

# Stop and remove Docker containers
down:
    $(DOCKER_COMPOSE) down

# Restart Docker containers
restart: down up

# Clean Docker containers and volumes
clean:
    $(DOCKER_COMPOSE) down -v --remove-orphans

# Show logs from Docker containers
logs:
    $(DOCKER_COMPOSE) logs -f

# Run tests inside Docker container
test:
    $(DOCKER_COMPOSE) run --rm app poetry run pytest

# Shell into Docker container
shell:
    $(DOCKER_COMPOSE) run --rm app bash

# Help target
help:
    @echo "Usage:"
    @echo "  make build          - Build Docker containers"
    @echo "  make up             - Run Docker containers in detached mode"
    @echo "  make down           - Stop and remove Docker containers"
    @echo "  make restart        - Restart Docker containers"
    @echo "  make clean          - Clean Docker containers and volumes"
    @echo "  make logs           - Show logs from Docker containers"
    @echo "  make test           - Run tests inside Docker container"
    @echo "  make shell          - Shell into Docker container"
    @echo "  make help           - Show this help message"

.PHONY: all build up down restart clean logs test shell help