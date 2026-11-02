FROM python:3.11-slim

# Create a non-root user for security
RUN useradd -m appuser

WORKDIR /app

COPY requirements.txt .
# Use --only-binary :all: to prevent execution of setup scripts from source packages
RUN pip install --no-cache-dir --only-binary :all: -r requirements.txt

COPY . .

# Set ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
