import os
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

DATA_DIR = "data"
VECTOR_DIR = "vectors"

# ----------------------------
# ONE MAJOR CITY PER STATE
# ----------------------------
MAJOR_CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Kolkata", "Chennai", "Hyderabad",
    "Ahmedabad", "Pune", "Jaipur", "Lucknow", "Bhopal", "Patna",
    "Guwahati", "Chandigarh", "Bhubaneswar", "Thiruvananthapuram",
    "Ranchi", "Imphal", "Gangtok", "Kohima", "Shillong", "Agartala",
    "Aizawl", "Panaji", "Shimla", "Dehradun", "Srinagar", "Puducherry"
]

# ----------------------------
# LOAD TEXT FILES
# ----------------------------

def load_text_file(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"{filename} missing")
        return []
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = filename
    return docs

# ----------------------------
# LOAD CSV WITH CITY FILTERING
# ----------------------------

def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"{filename} missing")
        return []

    df = pd.read_csv(path)

    # If city/state column exists → filter to major cities only
    city_cols = [c for c in df.columns if "city" in c.lower() or "location" in c.lower() or "district" in c.lower()]

    if city_cols:
        city_col = city_cols[0]

        df_filtered = df[df[city_col].astype(str).str.contains("|".join(MAJOR_CITIES), case=False, na=False)]

        if len(df_filtered) == 0:
            df = df.head(300)
        else:
            df = df_filtered
    else:
        # dataset has no city/state → just sample
        df = df.head(300)

    docs = []
    for _, row in df.iterrows():
        text = " | ".join(f"{col}: {val}" for col, val in row.items())
        docs.append(Document(page_content=text, metadata={"source": filename}))

    return docs

# ----------------------------
# MAIN BUILD FUNCTION
# ----------------------------

def main():
    docs = []

    # 1. Subsidies (keep full)
    docs += load_text_file("subsidy.txt")

    # 2. Crop & Soil Data (full dataset)
    docs += load_csv("data_core.csv")

    # 3. Mandi data → filter by major cities
    docs += load_csv("mandi_data_2000_rows.csv")

    # 4. Rainfall dataset → filtered automatically
    docs += load_csv("Indian Rainfall Dataset District-wise Daily Measurements.csv")

    print("Loaded documents:", len(docs))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    print("Chunks:", len(chunks))

    embeddings = OllamaEmbeddings(
        model="llama3.2:3b",
        base_url="http://localhost:11434"
    )

    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=VECTOR_DIR
    )
    vectordb.persist()

    print("Optimized RAG Vector DB built successfully!")

if __name__ == "__main__":
    os.makedirs(VECTOR_DIR, exist_ok=True)
    main()
