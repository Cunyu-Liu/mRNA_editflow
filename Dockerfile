# Clean-environment reproduction image for mrna_editflow (P0-06).
#
# Build:  docker build -t mrna_editflow .
# Run:    docker run --rm mrna_editflow
#
# The build itself executes the P0-06 acceptance gate:
#   pip install -e . && pytest -q && bash scripts/reproduce_smoke.sh
# so a successful build IS the passing acceptance.

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- dependencies first (layer caching) ---
COPY pyproject.toml requirements-lock.txt ./
# CPU torch wheel (keeps the image small; no CUDA in the acceptance gate).
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements-lock.txt

# --- source ---
COPY . .

# --- acceptance gate ---
RUN pip install -e .
RUN pytest -q
RUN bash scripts/reproduce_smoke.sh

CMD ["bash"]
