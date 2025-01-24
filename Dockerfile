# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements-linux.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable

# Remove `parallel_tool_calls=False` argument from langchain_openai
RUN sed -i 's/, parallel_tool_calls=False//g' /usr/local/lib/python3.10/site-packages/langchain_openai/chat_models/base.py

# Run streamlit when the container launches, also enable development features like auto-reload
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]
