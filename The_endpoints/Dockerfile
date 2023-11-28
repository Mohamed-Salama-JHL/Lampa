# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install fastapi uvicorn

RUN apt-get update && apt-get install -y wget && apt-get clean

# Expose the port number the app runs on
EXPOSE 7000

#CMD ["uvicorn", "map_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "-m", "uvicorn", "APIs.main:app", "--host", "0.0.0.0", "--port", "7000"]
