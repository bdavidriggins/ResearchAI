#Build and run commands
# docker buildx build --tag research_ai_project .
# docker run --network=host -p 5000:5000 research_ai_project


# Use the official slim version of Python 3.12 as the base image, keeping the image lightweight
FROM python:3.11-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Install necessary system dependencies required for building certain Python packages
RUN apt-get update && apt-get install -y \
    swig \                  
    build-essential \        
    cmake \
    libopenblas-dev \
    libomp-dev \
    libatlas-base-dev \
    git \
    && apt-get clean

# Copy the Python dependencies file into the container
COPY requirements.txt .

# Modify requirements.txt to use the pre-built faiss-cpu package for better compatibility and faster setup
RUN sed -i 's/faiss-cpu==1.7.4/faiss-cpu/g' requirements.txt

# Install the required Python dependencies without caching the packages to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code into the working directory in the container
COPY . .

# Expose port 5000 to allow external access to the service running inside the container
EXPOSE 5000

# Set the default command to run the application using Python
CMD ["python", "app.py"]
