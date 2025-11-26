# Base Image (Python 3.10 Slim for smaller size)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# libsndfile1 is required for soundfile/librosa
# ffmpeg is required for audio processing
# gcc/python3-dev are needed for building some python extensions
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port used by Uvicorn
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
