test:
	echo; echo; echo "Unit tests:"; coverage run --source=deconvolution -m unittest discover -s tests; echo; echo; echo "Code coverage report: "; coverage report

.PHONY: test
