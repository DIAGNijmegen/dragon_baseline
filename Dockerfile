FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

RUN apt-get -y update
RUN apt-get -y install git

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools


# Install the requirements
COPY --chown=user:user requirements.txt .
RUN python -m pip install --user -r requirements.txt \
    && rm -rf /home/user/.cache/pip

# Download the model, tokenizer and metrics
COPY --chown=user:user download_model.py .
RUN python download_model.py --model_name distilbert-base-multilingual-cased
COPY --chown=user:user download_metrics.py .
RUN python download_metrics.py

# Set the environment variables
ENV TRANSFORMERS_OFFLINE=1
ENV HF_EVALUATE_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Install the app
RUN mkdir -p /opt/app/dragon_baseline
COPY --chown=user:user . /opt/app/dragon_baseline
RUN python -m pip install --user /opt/app/dragon_baseline

COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
