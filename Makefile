####################
# setup

NAME=elapid
CONDA=conda run --name ${NAME}

# help docs
.DEFAULT: help
help:
	@echo "--- [ $(NAME) developer tools ] --- "
	@echo ""
	@echo "make init        - initialize conda dev environment"
	@echo "make test        -	run package tests"
	@echo "make test-data   - generates new data for tests/data/"
	@echo "make ci          - regenerates CI recipe"
	@echo "make conda-clean - removes conda tempfiles"

####################
# utils

init:
	conda env list | grep -q ${NAME} || conda create --name=${NAME} python=3.7 -y
	${CONDA} conda install -c conda-forge mamba -y
	${CONDA} pip install -r requirements-dev.txt
	${CONDA} python recipe/convert-dependency-format.py
	${CONDA} mamba install --file recipe/environment.yml -c conda-forge -y || exit 1
	${CONDA} pip install -e .
	${CONDA} pre-commit install || exit 1
	rm -f recipe/environment.yml

test:
	${CONDA} pytest --cov --no-cov-on-fail --cov-report=term-missing:skip-covered

test-data:
	${CONDA} python tests/create_test_data.py

conda-clean:
	conda clean --all

ci:
	conda smithy rerender -c auto
