# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# System dependencies (for building some Python libs)
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install required Python packages
RUN pip install -r requirements.txt

# Set environment variable (used for runtime or passed via --env or .env file)
ENV GROQ_API_KEY=${GROQ_API_KEY}

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit app (no telemetry, internal port 8501, accessible from container IP)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
