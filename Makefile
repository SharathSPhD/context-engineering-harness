.PHONY: test test-unit test-integration coverage install reproduce-h1 reproduce-h2 reproduce-h3 reproduce-h4 reproduce-h5 reproduce-h6 reproduce-h7 validate validate-fast validate-h1 validate-h2 validate-h3 validate-h4 validate-h5 validate-h6 validate-h7 report

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

# ── Validation suite (claude CLI subscription auth, no ANTHROPIC_API_KEY needed) ──

validate:
	.venv/bin/python experiments/validate/runner.py
	.venv/bin/python experiments/validate/report.py

validate-fast:
	.venv/bin/python experiments/validate/runner.py --skip-llm
	.venv/bin/python experiments/validate/report.py

validate-h1:
	.venv/bin/python experiments/validate/h1_schema.py

validate-h2:
	.venv/bin/python experiments/validate/h2_rag.py

validate-h3:
	.venv/bin/python experiments/validate/h3_agents.py

validate-h4:
	.venv/bin/python experiments/validate/h4_compaction.py

validate-h5:
	.venv/bin/python experiments/validate/h5_multiagent.py

validate-h6:
	.venv/bin/python experiments/validate/h6_classifier.py

validate-h7:
	.venv/bin/python experiments/validate/h7_forgetting.py

report:
	.venv/bin/python experiments/validate/report.py
