import pathlib
import pickle

from vectorize import Vectorizer


print("Vectorizer Initialize")
vectorizer = Vectorizer()

print("Images Embedding")
embbed = vectorizer.pre_image_embedding(pathlib.Path("Images"))
print(f"{len(embbed)} images embedd")

with open("data/images_embed.pkl", "wb") as f:
    pickle.dump(embbed, f)
print("saved images")
