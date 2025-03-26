# BDC Chatbot (WIP)

Chatbot for Biodata Catalyst

## Get documentation
Clone the [BDC website repo](https://github.com/stagecc/interim-bdc-website/tree/main) to a directory with the same parent directory as this repo.

## Copy environment variables
Copy `.env_example` to `.env` and make any necessary changes

## Install dependencies

```bash
pip install -r requirements.txt
```

## Port forwarding for local development using vLLM and Ollama at RENCI

```bash
kubectl -n ner port-forward svc/vllm-llama-3-1-8b-instruct 8080:80  # vLLM port-forward

kubectl -n ner port-forward svc/ollama 11434:11434                  # Ollama port-forward
```
Edit `.env` to match

## Create RAG database

```bash
python -m utils.prepare_preproc_doc                 # preprocess BDC website repo
python -m utils.prepare_chromadb                    # create chroma db
```

## Run chatbot server

```bash
fastapi dev server.py
```

## Client

See [renci/bdc_chatbot-client-streamlit](https://github.com/RENCI/BDC_Chatbot-client-streamlit).
