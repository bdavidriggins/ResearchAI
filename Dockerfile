# Use the Python 3.12-slim base image
FROM python:3.12-slim

# Set the working directory to /app
WORKDIR /app

# Install required system dependencies for other packages
RUN apt-get update && apt-get install -y \
    swig \
    build-essential \
    cmake \
    libopenblas-dev \
    libomp-dev \
    libatlas-base-dev \
    git \
    && apt-get clean

# Copy requirements.txt into the container
COPY requirements.txt .

# Replace faiss-cpu source install with the pre-built version
RUN sed -i 's/faiss-cpu==1.7.4/faiss-cpu/g' requirements.txt

# Install the required dependencies globally
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 5000
EXPOSE 5000

# Set the default command to run the application
CMD ["python", "app.py"]
