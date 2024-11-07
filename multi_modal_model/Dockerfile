# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into the container at /app
COPY . .

# Make run_all.sh executable
RUN chmod +x run_all.sh

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run run_all.sh when the container launches
CMD ["./run_all.sh"]
