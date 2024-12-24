# Use a Python version compatible with most libraries (Python 3.12)
FROM python:3.12-slim

# Set working directory
WORKDIR /NEO4J_RAG

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --upgrade -r requirements.txt

# Copy application files
COPY . .

# Expose the port used by Streamlit
EXPOSE 8503

# Add a health check for Docker
HEALTHCHECK CMD curl --fail http://localhost:8503/_stcore/health || exit 1

# Set the entry point for the application
ENTRYPOINT ["streamlit", "run", "bot.py", "--server.port=8503", "--server.address=0.0.0.0"]
