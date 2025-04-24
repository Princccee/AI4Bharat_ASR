# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y libgomp1
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 5000 for Flask app
EXPOSE 5000

# Set environment variable to indicate Flask is in production mode
ENV FLASK_ENV=production

# Run the application
CMD ["python", "app.py"]
