# AI-Based Data Quality Monitoring System
# Default target runs the FastAPI service. Override CMD to run the Streamlit
# dashboard instead (see README > Docker section) -- both share this image.

FROM python:3.12-slim

LABEL maintainer="AI Data Quality Monitoring System" \
      description="Production-ready data quality monitoring: validation, ML classification, anomaly detection, alerting"

WORKDIR /app

# System deps kept minimal: build-essential is needed for a couple of
# scientific-python wheels that don't ship manylinux binaries on every arch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user for defense-in-depth
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Default: FastAPI service. To run the dashboard instead:
#   docker run -p 8501:8501 <image> streamlit run dashboard/dashboard.py --server.address 0.0.0.0
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
