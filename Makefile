.PHONY: test cov cov-html diff fmt lint

PYTEST=pytest -q --maxfail=1 --disable-warnings
COVOPTS=--cov --cov-report=term-missing:skip-covered

test:
	$(PYTEST) tests

cov:
	$(PYTEST) $(COVOPTS) --cov-report=xml:coverage.xml

cov-html:
	$(PYTEST) $(COVOPTS) --cov-report=html

# Порівняння diff coverage відносно main (локально)
diff: cov
	diff-cover coverage.xml --compare-branch origin/main --fail-under=100
