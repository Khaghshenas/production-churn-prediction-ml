FROM python:3.11-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# The working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src/ ./src/
COPY models/ ./models/
COPY config.yaml .

# Create folders that the app expects to exist
RUN mkdir -p data models

# Expose the port
EXPOSE 8000

# Run the API using uvicorn
# We use 0.0.0.0 to allow connections from outside the container
CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]