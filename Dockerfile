# Use an official Python runtime as a parent image
FROM python:3.12

# get website source files
RUN git clone https://github.com/stagecc/interim-bdc-website

# Set the working directory in the container to /app
WORKDIR /app

# Declare build-time arguments
ARG DB_PATH
ARG OPENAI_API_KEY

# Set runtime environment variables
ENV DB_PATH=$DB_PATH
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# Copy the current directory contents into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# add Comparator.IN, Comparator.NIN to chroma.py
RUN sed -i '/allowed_comparators = \[/a\        Comparator.IN,\n        Comparator.NIN,' /usr/local/lib/python3.12/site-packages/langchain_community/query_constructors/chroma.py

# Remove `parallel_tool_calls=False` argument from langchain_openai
RUN sed -i 's/parallel_tool_calls=False,//g' /usr/local/lib/python3.12/site-packages/langchain_openai/chat_models/base.py

COPY . .

# create RAG database
RUN python ./utils/preproc_doc.py && python -m utils.prepare_chromadb

EXPOSE 8501

# Run streamlit when the container launches, also enable development features like auto-reload
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
