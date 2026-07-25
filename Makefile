.DEFAULT_GOAL := help
SHELL := /bin/bash
PY := backend/.venv/bin/python
PIP := backend/.venv/bin/pip

.PHONY: help install install-backend install-frontend dev backend frontend build check typecheck clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: install-backend install-frontend ## Install both toolchains

install-backend: ## Create the Python venv and install requirements
	python3 -m venv backend/.venv
	$(PIP) install --upgrade pip
	$(PIP) install -r backend/requirements.txt

install-frontend: ## Install npm dependencies
	cd frontend && npm install

backend: ## Run the FastAPI server with reload
	cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000

frontend: ## Run the Vite dev server
	cd frontend && npm run dev

dev: ## Run backend and frontend together
	@$(MAKE) -j2 backend frontend

build: ## Production build of the frontend
	cd frontend && npm run build

typecheck: ## TypeScript project check
	cd frontend && npx tsc -b --force

check: typecheck ## Typecheck plus backend import smoke test
	cd backend && .venv/bin/python -c "import app.main; print('backend ok')"

clean: ## Remove build output and caches
	rm -rf frontend/dist frontend/node_modules/.tmp backend/.data
	find backend -name __pycache__ -type d -prune -exec rm -rf {} +
