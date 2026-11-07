# Stage 1: Build stage
FROM python:3.11-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev

WORKDIR /build

# Use the lock file for deterministic, secure installation
COPY requirements.lock .
RUN pip install --no-cache-dir --user --only-binary :all: -r requirements.lock

# Stage 2: Runtime stage
FROM python:3.11-slim-bookworm

# Create a non-privileged user for security
RUN useradd -m appuser

WORKDIR /app

# Copy only the installed packages from the builder stage
COPY --from=builder /root/.local /home/appuser/.local
COPY . .

# Ensure appuser owns the files and update PATH to include the local bin
RUN chown -R appuser:appuser /app
USER appuser
ENV PATH=/home/appuser/.local/bin:${PATH}

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
