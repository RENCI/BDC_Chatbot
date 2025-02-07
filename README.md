# BDC Chatbot (WIP)

Chatbot for Biodata Catalyst

## Development

### Get documentation
Clone the [BDC website repo](https://github.com/stagecc/interim-bdc-website/tree/main) to a directory with the same parent directory as this repo.

### Copy environment variables
Copy `.env_example` to `.env` and make any necessary changes

### Install dependencies

```bash
pip install -r requirements.txt
```

### Port forwarding for local development using vLLM and Ollama at RENCI

```bash
kubectl -n ner port-forward svc/vllm-llama-3-1-8b-instruct 8080:80  # vLLM port-forward

kubectl -n ner port-forward svc/ollama 11434:11434                  # Ollama port-forward
# kubectl -n ner port-forward ollama-dd6845745-2txz8 11434:11434
```
Edit `.env` to match

### Create RAG database

```bash
python ./src/preproc_doc.py                                         # preprocess BDC website repo
python -m src.prepare_chromadb                                      # create chroma db
```

### Run chatbot
```bash
streamlit run app.py                                # run streamlit app
```

## Docker

Like above, you'll need to ensure an `.env` file is in place with these two variables defined:
```
DB_PATH=./chroma_db
OPENAI_API_KEY=sk-proj-321321
```

Build the image.

```bash
$ docker build --build-arg DB_PATH=$(grep DB_PATH .env | cut -d '=' -f2) \
               --build-arg OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d '=' -f2) \
               -t containers.renci.org/comms/bdc-chatbot:0.1.0 .

```

Verify the image works.

```bash
$ docker run --rm --name bdc-chatbot -p 8501:8501 containers.renci.org/comms/bdc-chatbot:0.1.0
```

Push to container registry.
```bash
docker push containers.renci.org/comms/bdc-chatbot:0.1.0
```

Update version tag in `chart/values.yaml`.
```
image:
  tag: "0.1.0" # match tag to version
```

Deploy.
```
helm install bdcbot ./chart/ -n comms
```

## Additional Notes

> [!WARNING]
>
> - If you see `ValueError: Received disallowed comparator nin` when running the chatbot app, add `Comparator.IN, Comparator.NIN` to `langchain_community\query_constructors\chroma.py` under `allowed_comparators`

> [!IMPORTANT]
>
> - To use ${\color{orange}\text{vLLM}}$ API for chat completion, remove `parallel_tool_calls=False` from `langchain_openai\chat_models\base.py`
> - Chroma DB initialization might quit without error or warning, might be caused by compatibility issue with Windows.
