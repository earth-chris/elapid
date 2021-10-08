# environment variables
NAME=elapid
CONDA_RUN=conda run --name ${NAME}
DOCKER_IMAGE=us.gcr.io/california-fores-1547767414612/$(NAME)
DOCKER_TAG_GIT:=$(shell git rev-parse --short=12 HEAD)

# help docs
.DEFAULT: help
help:
	@echo "--- [ $(NAME) ] --- "
	@echo ""

	@echo "make init"
	@echo "	initialize development tools -- conda, pip, etc"

	@echo "make update"
	@echo "	update development tools"

	@echo "make test"
	@echo "	run tests"

	@echo "make test-data"
	@echo " generates new data for tests/data/"


# paths for dummy files to indicate when targets need to be rerun
DIR_MAKE=.make
CONDA_UPDATED=${DIR_MAKE}/conda_updated
PIP_UPDATED=${DIR_MAKE}/pip_updated
DOCKER_INITIALIZED=${DIR_MAKE}/docker_initialized


####################
# ENTRY POINTS

init: conda-init pip-init misc-update
	@:

update: conda-update pip-update misc-update
	@:

test:
	${CONDA_RUN} pytest --cov --no-cov-on-fail --cov-report=term-missing:skip-covered

test-data:
	${CONDA_RUN} python tests/create_test_data.py

# conda
conda-init:
	@conda env list | grep -q -w ${CONDA_ENV} || conda env create --file environment.yml
	@test -d ${DIR_MAKE} || mkdir ${DIR_MAKE}
	@touch ${CONDA_UPDATED}

conda-update: conda-init ${CONDA_UPDATED}
	@:
${CONDA_UPDATED}: environment.yml
	${CONDA_RUN} conda env update --file environment.yml --prune
	@test -d ${DIR_MAKE} || mkdir ${DIR_MAKE}
	@touch ${CONDA_UPDATED}

conda-clean:
	@conda clean --all


# pip
pip-init: ${PIP_UPDATED}
	@:
pip-update: ${PIP_UPDATED}
	@:
${PIP_UPDATED}: requirements.txt
	${CONDA_RUN} pip install -r requirements.txt
	${CONDA_RUN} pip install --editable .
	@test -d ${DIR_MAKE} || mkdir ${DIR_MAKE}
	@touch ${PIP_UPDATED}


# misc
misc-update:
	@${CONDA_RUN} pre-commit install
	@test ! -f .gcloudignore || rm .gcloudignore
	@cp .gitignore .gcloudignore
