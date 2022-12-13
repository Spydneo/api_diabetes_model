####### Base Image #######
FROM python:3.9 as python-base

ENV PYTHONUNBUFFERED=1 
ENV PYTHONDONTWRITEBYTECODE=1 
ENV PIP_NO_CACHE_DIR=off 
ENV PIP_DISABLE_PIP_VERSION_CHECK=on 
ENV PIP_DEFAULT_TIMEOUT=100 
ENV POETRY_VERSION=1.1.7 
ENV POETRY_HOME="/opt/poetry" 
ENV POETRY_VIRTUALENVS_IN_PROJECT=true 
ENV POETRY_NO_INTERACTION=1 
ENV PYSETUP_PATH="/opt/pysetup" 
ENV VENV_PATH="/opt/pysetup/.venv"

# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

########### Builder Image #############
FROM python-base as builder-base
RUN apt-get update && apt-get install --no-install-recommends -y curl build-essential

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://install.python-poetry.org | python3 -

# copy project requirement files here to ensure they will be cached.
WORKDIR $PYSETUP_PATH
COPY poetry.lock pyproject.toml ./

# TEst fix error
RUN curl https://bootstrap.pypa.io/ez_setup.py | python

# install runtime deps - uses $POETRY_VIRTUALENVS_IN_PROJECT internally
RUN poetry install --no-dev

################ Production Image ###########
FROM python-base as production
COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH
COPY api_diabetes_model /api_diabetes_model/
# COPY model /model/
RUN pwd && ls
RUN echo $(ls)
# RUN python -m model
WORKDIR /api_diabetes_model
EXPOSE 5000
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]