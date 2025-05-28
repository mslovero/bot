.PHONY: run clean

PYTHON := python3
RAG_SCRIPT := mi_rag.py
APP_SCRIPT := app.py
LOG_DIR := logs

run:
	@echo "🔵 Iniciando mi_rag.py en segundo plano..."
	@mkdir -p $(LOG_DIR)
	@$(PYTHON) $(RAG_SCRIPT) > $(LOG_DIR)/mi_rag.log 2>&1 &
	@echo "🔵 Iniciando app.py..."
	@$(PYTHON) $(APP_SCRIPT) 2>&1 | tee $(LOG_DIR)/app.log
	@echo "🟢 Procesos terminados."

stop:
	@pkill -f $(RAG_SCRIPT) || true
	@echo "🔴 Procesos detenidos."

clean:
	@rm -rf $(LOG_DIR)
	@echo "🧹 Logs eliminados."