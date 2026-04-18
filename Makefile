.PHONY: test test-unit test-integration coverage install reproduce-h1 reproduce-h2 reproduce-h3 reproduce-h4 reproduce-h5 reproduce-h6 reproduce-h7

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/ -v --tb=short -m "not integration"

test-integration:
	pytest tests/ -v --tb=short -m integration

coverage:
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

reproduce-h1:
	RANDOM_SEED=42 python experiments/h1_schema_congruence/run.py

reproduce-h2:
	RANDOM_SEED=42 python experiments/h2_precision_rag/run.py

reproduce-h3:
	RANDOM_SEED=42 python experiments/h3_buddhi_manas/run.py

reproduce-h4:
	RANDOM_SEED=42 python experiments/h4_event_boundary/run.py

reproduce-h5:
	RANDOM_SEED=42 python experiments/h5_avacchedaka_multiagent/run.py

reproduce-h6:
	RANDOM_SEED=42 python experiments/h6_khyativada_classifier/run.py

reproduce-h7:
	RANDOM_SEED=42 python experiments/h7_adaptive_forgetting/run.py

annotate-h6:
	python experiments/h6_khyativada_classifier/annotate.py

build-benchmarks:
	python experiments/h1_schema_congruence/build_benchmark.py
	python experiments/h2_precision_rag/build_benchmark.py
