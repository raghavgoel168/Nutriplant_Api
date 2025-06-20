# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from your project into the container
COPY . .

# Upgrade pip and install dependencies without cache
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 5000

# Start the app using Gunicorn (better than Flask's built-in server)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
