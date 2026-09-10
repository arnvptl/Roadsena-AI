FROM python:3.12-slim

# Install system libraries required by opencv-python-headless and reportlab
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port (Railway injects $PORT at runtime)
EXPOSE 8080

# Start with gunicorn
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
