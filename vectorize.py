"""
Vectorizer
"""
import os
import re
import json
import pathlib
import numpy as np
import torch

from typing import List, Union

import html
import ftfy

from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, BatchFeature
from tqdm import tqdm

from huggingface_hub import snapshot_download


class Vectorizer:
    def __init__(
        self,
        model_path: str = "/app/models",
        dim: int = 768,
        hugging_face_token=None,
    ):
        hugging_face_token = hugging_face_token or os.environ.get(
            "HUGGING_FACE_TOKEN",
        )

        self.dim = dim  # Embedding Vector Length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # CLIP
        self.clip = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_safetensors=True,
        ).to(self.device)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
        )

        # Image Processor
        self.processor = AutoImageProcessor.from_pretrained(
            model_path,
        )

    def basic_clean(self, text: str) -> str:
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def whitespace_clean(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    # embedding data for Initialize VectorDB
    def initialize(
        self,
        texts: Union[str, List[str]],
        max_seq_len: int = 77,
    ):
        # Initialize Texts
        features = self.tokenize(texts, max_seq_len).to(self.device)
        with torch.no_grad():
            vecs = self.clip.get_text_features(**features).to("cpu")
        meili_list = []
        for id, text in enumerate(texts):
            meili_list.append(
                {
                    "id": id,
                    "_vectors": vecs[id].detach().numpy().copy(),
                    "text": text,
                }
            )
        return meili_list

    # tokenize multiple sentences
    def tokenize(
        self,
        texts: Union[str, List[str]],
        max_seq_len: int = 77,
    ) -> BatchFeature:
        if isinstance(texts, str):
            texts = [texts]
        texts = [self.whitespace_clean(text) for text in texts]

        inputs = self.tokenizer(
            texts,
            max_length=max_seq_len - 1,
            padding="max_length",
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = [[self.tokenizer.bos_token_id] + ids for ids in inputs["input_ids"]]
        attention_mask = [[1] + am for am in inputs["attention_mask"]]
        position_ids = [list(range(0, len(input_ids[0])))] * len(texts)

        return BatchFeature(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(
                    attention_mask,
                    dtype=torch.long,
                ),
                "position_ids": torch.tensor(position_ids, dtype=torch.long),
            }
        )

    # embedding images and formatting for meilisearch
    def pre_image_embedding(
        self,
        image_dir: pathlib.Path,
        batch_size: int = 256,
    ):
        # get images path
        image_paths = list(image_dir.glob("**/*.jpg"))

        # Split images per batch size
        batches = [
            image_paths[i : i + batch_size]
            for i in range(0, len(image_paths), batch_size)
        ]

        all_embeddings = []
        image_id = 0

        # Embedding each batches
        for batch in tqdm(batches):
            images = [Image.open(image_path) for image_path in batch]
            embeddings = self.image_vectorize(images)
            for image_path, vec in zip(batch, embeddings):
                all_embeddings.append(
                    {"id": image_id, "path": str(image_path), "_vectors": vec}
                )
                image_id += 1  # increment id

        return all_embeddings

    def text_vectorize(
        self,
        text: str,
        max_seq_len: int = 77,
    ) -> List:
        tokens = self.tokenize([text], max_seq_len).to(self.device)
        with torch.no_grad():
            vec = self.clip.get_text_features(**tokens).to("cpu")
            return vec.detach().numpy().copy()[0]

    def image_vectorize(self, images: list) -> list:
        processed_images = self.processor(
            images=images,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            vec = self.clip.get_image_features(**processed_images).to("cpu")
            return vec.detach().numpy().copy()


if __name__ == "__main__":
    print("Initialize Vectorize")
    vectorizer = Vectorizer()
    device = vectorizer.device
    print("Initialized Vectorized")
    print("basic_clean")
    print(f"{vectorizer.basic_clean('赤い リンゴ')}")
    print("whitespace_clean")
    print(f"{vectorizer.basic_clean('赤い リンゴ')}")
    print("tokenize")
    res = vectorizer.tokenize("赤いりんご")
    print(f"tokenized object type: {type(res)}")
    print("Initial Tokenize")
    res = vectorizer.initialize(
        ["これは赤い橋です", "それは青い箸です", "あれは黄色い花です"],
    )
    print(res)

    print(vectorizer.text_vectorize("私の名前は日本太郎です。").dtype)
    print(vectorizer.text_vectorize("私の名前は日本太郎です。").shape)

    print("embedding image")
    image0 = Image.open("Images/667626_18933d713e.jpg")
    print(vectorizer.image_vectorize(images=[image0]).dtype)
    print(vectorizer.image_vectorize(images=[image0]).squeeze().shape)
    image1 = Image.open("Images/3637013_c675de7705.jpg")
    image2 = Image.open("Images/70995350_75d0698839.jpg")
    print(len(vectorizer.image_vectorize(images=[image1, image2])))
    print(vectorizer.image_vectorize(images=[image1, image2])[0].shape)
