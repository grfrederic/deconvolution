# Run unit tests
test:
	python -m unittest discover -s tests

# Check test coverage using unittest module
coverage:
	coverage run --source=deconvolution -m unittest discover -s tests; coverage report

html:
	coverage run --source=deconvolution -m unittest discover -s tests; coverage html; python -m webbrowser "./htmlcov/index.html" &

.PHONY: test
