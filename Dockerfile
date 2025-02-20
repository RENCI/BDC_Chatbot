FROM python:3.12-slim AS builder

# set working directory
WORKDIR /app

# Install git temporarily for cloning, then remove it
RUN apt-get update && apt-get install -y git && \
    git clone --depth=1 https://github.com/stagecc/interim-bdc-website temp_repo && \
    cp -r temp_repo/* . && \
    rm -rf temp_repo && \
    apt-get remove -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# copy only requirements file to leverage caching
COPY requirements.txt .

# install dependencies in a temporary layer
RUN pip install --no-cache-dir -r requirements.txt --target /dependencies

# modify langchain files before finalizing the build
RUN sed -i '/allowed_comparators = \[/a\        Comparator.IN,\n        Comparator.NIN,' /dependencies/langchain_community/query_constructors/chroma.py && \
    sed -i 's/parallel_tool_calls=False,//g' /dependencies/langchain_openai/chat_models/base.py


# --- final stage ---
FROM python:3.12-slim AS runtime

# Install git temporarily for cloning, then remove it, again
RUN apt-get update && apt-get install -y git && \
    git clone --depth=1 https://github.com/stagecc/interim-bdc-website temp_repo && \
    cp -r temp_repo/* . && \
    rm -rf temp_repo && \
    apt-get remove -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# set working directory
WORKDIR /app

# declare build-time arguments
ARG DB_PATH
ARG OPENAI_API_KEY

# set runtime environment variables
ENV DB_PATH=$DB_PATH
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# copy only the necessary dependencies from the builder stage
COPY --from=builder /dependencies /usr/local/lib/python3.12/site-packages/

# copy application files
COPY . .

# create RAG database
RUN python ./utils/preproc_doc.py && python -m utils.prepare_chromadb

# expose Streamlit port
EXPOSE 8501

# run Streamlit when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
