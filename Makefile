.PHONY: install train test lint clean serve

install:
	pip install -e ".[dev]"

train:
	python3 scripts/train_specialist.py $(op) $(ARGS)

train-quick:
	python3 scripts/train_specialist.py add --quick

serve:
	tabula-rasa serve

test:
	python3 -m pytest tests/ -v --tb=short

test-quick:
	python3 -m pytest tests/test_property_based.py tests/test_tokenizer.py tests/test_config.py -v --tb=short

lint:
	ruff check src/tabula_rasa/ scripts/ tests/
	ruff format --check src/tabula_rasa/ scripts/ tests/

format:
	ruff format src/tabula_rasa/ scripts/ tests/
	ruff check --fix src/tabula_rasa/ scripts/ tests/

typecheck:
	mypy src/tabula_rasa/ --ignore-missing-imports || true

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
