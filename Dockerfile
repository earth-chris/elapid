FROM continuumio/miniconda:latest

# create working directory
ENV APP_HOME /app
WORKDIR $APP_HOME

# install the conda environment
COPY requirements.txt .
COPY environment.yml .
RUN conda env create --file environment.yml
