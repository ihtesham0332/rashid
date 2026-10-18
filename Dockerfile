# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Ensure Python output is unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files and model artifacts
COPY . .

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Launch API service
CMD ["python", "app.py"]
