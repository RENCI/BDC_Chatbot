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
# kubectl -n ner port-forward ollama-dd6845745-2txz8 11434:11434
```
Edit `.env` to match

## Create RAG database

```bash
python -m src.preproc_doc                                         # preprocess BDC website repo
python -m src.prepare_chromadb                                      # create chroma db
```

## Run chatbot
```bash
streamlit run ./src/streamlit_app.py                                # run streamlit app
```

> [!WARNING]
>
> - If you see `ValueError: Received disallowed comparator nin` when running the chatbot app, add `Comparator.IN, Comparator.NIN` to `langchain_community\query_constructors\chroma.py` under `allowed_comparators`

> [!IMPORTANT]
>
> - To use ${\color{orange}\text{vLLM}}$ API for chat completion, remove `parallel_tool_calls=False` from `langchain_openai\chat_models\base.py`
> - Chroma DB initialization might quit without error or warning, might be caused by compatibility issue with Windows.
