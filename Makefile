####################
# setup

NAME=elapid
CONDA=conda run --no-capture-output --name ${NAME}

# help docs
.DEFAULT: help
help:
	@echo "--- [ $(NAME) developer tools ] --- "
	@echo ""
	@echo "make init        - initialize conda dev environment"
	@echo "make test        - run package tests"
	@echo "make test-data   - generates new data for tests/data/"
	@echo "make conda-clean - removes conda tempfiles"
	@echo "make destroy     - deletes the $(NAME) conda env"

####################
# utils

init:
	conda env list | grep -q ${NAME} || conda create --name=${NAME} python=3.11 mamba -y -c conda-forge -c nodefaults
	${CONDA} mamba install gdal poetry -c conda-forge -c nodefaults -y
	${CONDA} poetry install

test:
	${CONDA} pytest -n auto --cov --no-cov-on-fail --cov-report=term-missing:skip-covered

test-data:
	${CONDA} python tests/create_test_data.py

conda-clean:
	conda clean --all

destroy:
	conda env remove -n ${NAME}
