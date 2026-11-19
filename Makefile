.PHONY: all data report test
all: data report
data:
	python scripts/generate_data.py
report:
	python -m ltv.pipeline
test:
	python -m pytest -q
