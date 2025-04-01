# Use an official lightweight Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Install required system libraries (fixes OpenCV OpenGL issue)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy required files
COPY requirements.txt .
COPY app.py .
COPY wsgi.py .
COPY model/ model/
COPY templates/ templates/
COPY static/ static/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables to avoid TensorFlow warnings
ENV TF_CPP_MIN_LOG_LEVEL=3

# Expose the default Hugging Face Spaces port
EXPOSE 7860

# Start the app using gunicorn with optimizations
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers=1", "--threads=2", "--timeout=120", "wsgi:application"]
