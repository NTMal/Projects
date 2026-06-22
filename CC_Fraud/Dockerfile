# ═══════════════════════════════════════════════════════════════════════════════
# Dockerfile — Recipe for building the CC Fraud Detection container image
#
# A Dockerfile is read top to bottom. Each line is a "layer".
# Docker caches layers — if nothing changed in a layer, it reuses the cache.
# This is why we copy requirements.txt BEFORE copying the rest of the code:
# dependencies change rarely, so they stay cached between builds.
#
# Build command (run from the CC_Fraud/ directory):
#   docker build -t cc-fraud-api .
#
# Run command:
#   docker run -p 8080:8080 cc-fraud-api
# ═══════════════════════════════════════════════════════════════════════════════


# ── Base image ────────────────────────────────────────────────────────────────
# We start FROM an official Python image. This gives us a Linux OS + Python
# pre-installed. We don't need to install Python ourselves.
#
# python:3.11-slim = Python 3.11 on a minimal Debian Linux
# 'slim' means it excludes unnecessary tools (compilers, docs, etc.)
# keeping the final image size small (~150MB vs ~900MB for the full image)
FROM python:3.11-slim

# ── Working directory ─────────────────────────────────────────────────────────
# WORKDIR sets the default directory inside the container.
# All subsequent commands (COPY, RUN, CMD) run from here.
# Think of it as `cd /app` — we're working inside /app.
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# Some Python packages need system-level C libraries to compile.
# We install them here using apt-get (Debian's package manager).
# --no-install-recommends keeps things lean by skipping optional extras.
# The `rm -rf /var/lib/apt/lists/*` cleans up the package cache afterwards
# to reduce image size.
RUN apt-get update && apt-get install -y \
    build-essential \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# COPY the requirements file into the container first (before any other code).
# This way, Docker caches the pip install step.
# If you only change app code (not requirements), Docker reuses this cache
# and rebuilds in seconds instead of minutes.
COPY requirements.txt .

# RUN executes a shell command during the image build.
# --no-cache-dir tells pip not to save downloaded packages (saves space).
# --upgrade pip ensures we have the latest pip version.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
# COPY <source on your machine> <destination in container>
# We copy the entire app/ directory into /app/ inside the container.
# This includes:
#   - main.py          (the FastAPI app)
#   - train.py         (not needed at runtime, but useful for reference)
#   - models/          (the .joblib model files trained locally)
COPY app/ .

# ── Port ──────────────────────────────────────────────────────────────────────
# EXPOSE tells Docker (and humans reading this file) that the container
# listens on port 8080. This is required by GCP Cloud Run.
# Note: EXPOSE alone doesn't publish the port — you still need -p 8080:8080
# when running locally, but Cloud Run handles this automatically.
EXPOSE 8080

# ── Startup command ───────────────────────────────────────────────────────────
# CMD defines what runs when the container starts.
# We're starting the uvicorn ASGI server with:
#   main:app       → run the `app` object in main.py
#   --host 0.0.0.0 → listen on all network interfaces (required in containers;
#                    'localhost' would only be visible inside the container itself)
#   --port 8080    → listen on port 8080 (Cloud Run default)
#   --workers 1    → one worker process (sufficient for demo; scale up in prod)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
