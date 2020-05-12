FROM continuumio/miniconda3

COPY . /
WORKDIR /

COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "-n", "myenv", "python", "dataset_collection_tool.py" "--inference-mode"]