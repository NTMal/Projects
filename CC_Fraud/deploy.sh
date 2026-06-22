#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# deploy.sh — Deploy CC Fraud Detection API to GCP Cloud Run
#
# This script walks through every step of deploying your Docker container
# to Google Cloud Platform's serverless container service (Cloud Run).
#
# Prerequisites:
#   1. Google Cloud account (free tier works)
#   2. gcloud CLI installed: https://cloud.google.com/sdk/docs/install
#   3. Docker Desktop running locally
#   4. You've already run: python app/train.py  (to generate .joblib files)
#
# Usage:
#   chmod +x deploy.sh    # make script executable (first time only)
#   ./deploy.sh           # run the full deployment
#
# Or run each section manually, one block at a time.
# ═══════════════════════════════════════════════════════════════════════════════

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit these variables before running

PROJECT_ID="your-gcp-project-id"        # Find this in GCP Console → top of page
REGION="europe-west2"                    # London region (closest to Oxford)
SERVICE_NAME="cc-fraud-api"              # Name your Cloud Run service
IMAGE_NAME="cc-fraud-api"               # Name for your Docker image
REPO_NAME="fraud-detection"             # Artifact Registry repository name

# Full image path in GCP Artifact Registry
# Format: REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/IMAGE_NAME
IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}"

echo "════════════════════════════════════════"
echo "CC Fraud API — GCP Cloud Run Deployment"
echo "════════════════════════════════════════"
echo "Project  : $PROJECT_ID"
echo "Region   : $REGION"
echo "Service  : $SERVICE_NAME"
echo "Image    : $IMAGE_PATH"
echo ""


# ── STEP 1: Authenticate with GCP ────────────────────────────────────────────
# This opens a browser window to log in with your Google account.
# Only needed once — gcloud remembers your credentials.
echo "── Step 1: Authenticate ────────────────────────────────────────────────"
gcloud auth login

# Set your project as the active project
gcloud config set project $PROJECT_ID
echo "✓ Authenticated and project set to: $PROJECT_ID"


# ── STEP 2: Enable required GCP APIs ─────────────────────────────────────────
# GCP services are disabled by default — you have to opt in.
# These three are needed for: building images, storing them, and running them.
echo ""
echo "── Step 2: Enable APIs ─────────────────────────────────────────────────"
gcloud services enable \
    artifactregistry.googleapis.com \  # stores your Docker images
    run.googleapis.com \               # runs your containers
    cloudbuild.googleapis.com          # builds images in the cloud (optional)
echo "✓ APIs enabled"


# ── STEP 3: Create Artifact Registry repository ───────────────────────────────
# Artifact Registry is GCP's Docker image store (like Docker Hub but on GCP).
# We create a repository to hold our fraud detection images.
echo ""
echo "── Step 3: Create Artifact Registry repo ───────────────────────────────"
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \        # we're storing Docker images
    --location=$REGION \
    --description="CC Fraud Detection models"
echo "✓ Repository created: $REPO_NAME"


# ── STEP 4: Configure Docker to authenticate with GCP ────────────────────────
# This tells Docker to use your gcloud credentials when pushing to GCP.
# You only need to run this once per machine.
echo ""
echo "── Step 4: Configure Docker auth ──────────────────────────────────────"
gcloud auth configure-docker ${REGION}-docker.pkg.dev
echo "✓ Docker configured to authenticate with GCP"


# ── STEP 5: Build Docker image locally ───────────────────────────────────────
# This reads the Dockerfile and builds the image on your machine.
# The -t flag tags (names) the image so we can push it to GCP.
# The . at the end means "use the current directory as the build context"
echo ""
echo "── Step 5: Build Docker image ──────────────────────────────────────────"
echo "This may take 3-5 minutes the first time (downloading base image + installing packages)..."
docker build -t ${IMAGE_PATH}:latest .
echo "✓ Image built: ${IMAGE_PATH}:latest"


# ── STEP 6: Push image to GCP Artifact Registry ───────────────────────────────
# Upload the image from your local machine to GCP's image store.
# Cloud Run will pull it from here when deploying.
echo ""
echo "── Step 6: Push image to GCP ───────────────────────────────────────────"
echo "Uploading image (~500MB, may take a few minutes)..."
docker push ${IMAGE_PATH}:latest
echo "✓ Image pushed to Artifact Registry"


# ── STEP 7: Deploy to Cloud Run ───────────────────────────────────────────────
# This is the actual deployment command.
# Cloud Run creates a serverless container that:
#   - Scales to zero when not in use (no cost when idle)
#   - Scales up automatically when requests come in
#   - Gets a public HTTPS URL automatically
echo ""
echo "── Step 7: Deploy to Cloud Run ─────────────────────────────────────────"
gcloud run deploy $SERVICE_NAME \
    --image=${IMAGE_PATH}:latest \
    --platform=managed \           # managed = fully serverless (no infra to manage)
    --region=$REGION \
    --allow-unauthenticated \      # allow public access (needed for portfolio demo)
    --port=8080 \                  # must match EXPOSE in Dockerfile
    --memory=512Mi \               # 512MB RAM (sufficient for sklearn models)
    --cpu=1 \                      # 1 vCPU
    --min-instances=0 \            # scale to zero when idle (free tier friendly)
    --max-instances=3              # cap at 3 instances to control costs

echo ""
echo "✅ Deployment complete!"
echo ""


# ── STEP 8: Get the service URL ────────────────────────────────────────────────
# Cloud Run assigns a permanent HTTPS URL to your service.
# This is the URL you put on your CV and in your GitHub README.
echo "── Step 8: Get service URL ─────────────────────────────────────────────"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --platform=managed \
    --region=$REGION \
    --format='value(status.url)')

echo "✓ Your API is live at: $SERVICE_URL"
echo ""
echo "Try it:"
echo "  Health check : curl $SERVICE_URL/health"
echo "  API docs     : open $SERVICE_URL/docs   (paste in browser)"
echo ""
echo "Example fraud prediction:"
echo "  curl -X POST $SERVICE_URL/predict/combined \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"V1\": -1.35, \"V2\": -0.07, \"V3\": 2.53, \"V4\": 1.37, \"V5\": -0.33, \"V6\": 0.46, \"V7\": 0.23, \"V8\": 0.09, \"V9\": 0.36, \"V10\": 0.09, \"V11\": -0.55, \"V12\": -0.61, \"V13\": -0.99, \"V14\": -0.31, \"V15\": 1.46, \"V16\": -0.47, \"V17\": 0.20, \"V18\": 0.02, \"V19\": 0.40, \"V20\": 0.25, \"V21\": -0.01, \"V22\": 0.27, \"V23\": -0.11, \"V24\": 0.06, \"V25\": 0.12, \"V26\": -0.18, \"V27\": 0.13, \"V28\": -0.02, \"Amount\": 149.62}'"
