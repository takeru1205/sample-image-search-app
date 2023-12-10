# sample-image-search-app

Sample image search app in Japanse.

This repo is used [stabilityai/japanese-stable-clip-vit-l-16](https://huggingface.co/stabilityai/japanese-stable-clip-vit-l-16) and [faiss](https://github.com/facebookresearch/faiss) for vector search. The image data is used [Flickr 8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k/).

## Demo

https://github.com/takeru1205/sample-image-search-app/assets/20464437/bb6b3148-c308-4896-839d-8bc49e64db17




## Download Models

```python
import os
from huggingface_hub import snapshot_download

hugging_face_token = (
    hugging_face_token if None else os.environ.get("HUGGING_FACE_TOKEN")
)

snapshot_download(repo_id="stabilityai/japanese-stable-clip-vit-l-16", local_dir="models", token=hugging_face_token)
```

## Data pre embedding

```bash
$ make embed
```

## Run App

```bash
$ docker compose build
$ docker compose up -d
```
