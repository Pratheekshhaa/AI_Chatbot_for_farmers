import os
import numpy as np
from PIL import Image
from langchain_ollama import OllamaEmbeddings

DATASET_DIR = "data/plant_disease/test"
OUTPUT_DIR = "plant_vectors"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_embedding(path, embedder):
    try:
        img = Image.open(path).convert("RGB").resize((256, 256))
        emb = embedder.embed_image(img.tobytes())
        return np.array(emb)
    except:
        return None

def main():
    embedder = OllamaEmbeddings(
        model="llama3.2:3b",
        base_url="http://localhost:11434"
    )

    disease_vecs = {}

    for folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        print("Processing:", folder)
        vecs = []
        count = 0

        for file in os.listdir(folder_path):
            if file.lower().endswith(("jpg", "png", "jpeg")):
                img_path = os.path.join(folder_path, file)
                v = get_embedding(img_path, embedder)
                if v is not None:
                    vecs.append(v)
                    count += 1
                if count >= 5:
                    break

        if vecs:
            disease_vecs[folder] = np.mean(np.vstack(vecs), axis=0).tolist()

    np.save(os.path.join(OUTPUT_DIR, "disease_vectors.npy"), disease_vecs)
    print("Saved plant disease vectors!")

if __name__ == "__main__":
    main()
