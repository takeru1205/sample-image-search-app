import pickle
from logging import getLogger, basicConfig, INFO
from PIL import Image

import faiss
import numpy as np
import streamlit as st

from vectorize import Vectorizer

basicConfig(level=INFO)
logger = getLogger(__name__)

logger.info("load data")
# データの読み込み
with open("data/images_embed.pkl", "rb") as f:
    image_data = pickle.load(f)

embeddings = np.array([x["_vectors"] for x in image_data]).astype("float32")

logger.info("Indexing")
# FAISSインデックスの作成
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

vectorizer = Vectorizer()

logger.info("Start app!")
st.title("日本語 Multimodal Search")

# 検索ボックス
query = st.text_input("Enter your search query", key="search")


# 検索ボタン
if st.button("Search") or query:
    # クエリをベクトル化
    logger.info(f"input query: {query}")
    query_vector = vectorizer.text_vectorize(query)
    query_vector = np.array(query_vector).astype("float32")

    # FAISSで検索
    D, I = index.search(query_vector.reshape(1, -1), 10)  # 上位10件の結果

    logger.info(I[0])

    # 検索結果の表示
    for i, idx in enumerate(I[0]):
        image_info = image_data[idx]
        st.write("file: ", image_info["path"])
        st.image(
            Image.open(image_info["path"]),
            caption=f"Image ID: {image_info['id']}",
        )
