# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /code

# Copy the current directory contents into the container at /app
COPY requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
COPY . .
EXPOSE 5000
# Run app.py when the container launches
#CMD ["panel", "serve", "map_app/main.py", "--allow-websocket-origin=*"]
CMD ["python","-W", "ignore", "/code/map_app/main.py"]


