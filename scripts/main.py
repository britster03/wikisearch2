import os
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss


DATA_DIR = '../data'
EMBEDDINGS_DIR = '../embeddings'
FAISS_DIR = '../faiss'
METADATA_DIR = '../metadata'

INPUT_CSV = os.path.join(DATA_DIR, 'input.csv')  # path to csv
EMBEDDINGS_PKL = os.path.join(EMBEDDINGS_DIR, 'embeddings.pkl')  # path where embeds are saved 
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, 'faiss_index.idx')  # path to save faiss
METADATA_PKL = os.path.join(METADATA_DIR, 'metadata.pkl')  # path where metadata is saved
TEXT_COLUMNS = ['paragraph']  # list of columns in csv for which the embeds are to be generated
MODEL_NAME = 'all-MiniLM-L6-v2'  # sentence transformer
INDEX_TYPE = 'IndexFlatL2'  # type of faiss index to be generated 
TOP_K = 5  # top k similar vectors to search in 

def generate_embeddings(input_csv, text_columns, model_name):
    """
    Generates embeddings for specified text columns in a CSV and saves them as a pickle file.

    Args:
        input_csv (str): Path to the input CSV file.
        text_columns (list): List of column names containing text data.
        model_name (str): Pre-trained SentenceTransformer model name.

    Returns:
        dict: Dictionary containing embeddings for each specified column.
        list: Metadata list containing mappings from index to original data.
    """

    print("Loading CSV file...")
    df = pd.read_csv(input_csv)

    # checking if the columns given above exist in the csv
    for col in text_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the CSV file.")


    print(f"Loading the SentenceTransformer model '{model_name}'...")
    model = SentenceTransformer(model_name)

    embeddings_dict = {}
    metadata = []

    for col in text_columns:
        print(f"Generating embeddings for column: '{col}'")
        embeddings = []
        for text in tqdm(df[col].astype(str), desc=f"Embedding '{col}'"):
            embedding = model.encode(text, show_progress_bar=False)
            embeddings.append(embedding)
        embeddings_dict[col] = np.array(embeddings)

    # creating metadata mapping to original raw data
    print("Creating metadata...")
    metadata = df.to_dict(orient='records')

    return embeddings_dict, metadata

def save_embeddings(embeddings_dict, embeddings_pkl):
    """
    Saves embeddings dictionary to a pickle file.

    Args:
        embeddings_dict (dict): Dictionary containing embeddings.
        embeddings_pkl (str): Path to save the pickle file.
    """
    print(f"Saving embeddings to {embeddings_pkl}...")
    with open(embeddings_pkl, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    print("Embeddings successfully saved!")

def load_embeddings(embeddings_pkl):
    """
    Loads embeddings from a pickle file.

    Args:
        embeddings_pkl (str): Path to the pickle file.

    Returns:
        dict: Dictionary containing embeddings.
    """
    print(f"Loading embeddings from {embeddings_pkl}...")
    with open(embeddings_pkl, 'rb') as f:
        embeddings_dict = pickle.load(f)
    for col, emb in embeddings_dict.items():
        print(f"Loaded embeddings for column '{col}' with shape {emb.shape}")
    return embeddings_dict

def save_metadata(metadata, metadata_pkl):
    """
    Saves metadata to a pickle file.

    Args:
        metadata (list): List containing metadata mappings.
        metadata_pkl (str): Path to save the pickle file.
    """
    print(f"Saving metadata to {metadata_pkl}...")
    with open(metadata_pkl, 'wb') as f:
        pickle.dump(metadata, f)
    print("Metadata successfully saved!")

def load_metadata(metadata_pkl):
    """
    Loads metadata from a pickle file.

    Args:
        metadata_pkl (str): Path to the pickle file.

    Returns:
        list: Loaded metadata.
    """
    print(f"Loading metadata from {metadata_pkl}...")
    with open(metadata_pkl, 'rb') as f:
        metadata = pickle.load(f)
    print(f"Loaded metadata with {len(metadata)} records.")
    return metadata

def create_faiss_index(embeddings, index_type='IndexFlatL2'):
    """
    Creates a FAISS index from embeddings.

    Args:
        embeddings (np.ndarray): Numpy array of shape (num_vectors, dimension).
        index_type (str): Type of FAISS index to create.

    Returns:
        faiss.Index: The created FAISS index.
    """
    dimension = embeddings.shape[1]
    print(f"Creating FAISS Index of type '{index_type}' with dimension: {dimension}")

    if index_type == 'IndexFlatL2':
        index = faiss.IndexFlatL2(dimension)
    elif index_type == 'IVFFlat':
        nlist = 100  # number of clusters ot data
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        print("Training IVFFlat index...")
        index.train(embeddings)
    elif index_type == 'HNSW':
        index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is nums of neighbors
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    print("Adding embeddings to the FAISS index...")
    index.add(embeddings)
    print(f"Total vectors in the index: {index.ntotal}")

    return index

def save_faiss_index(index, index_path):
    """
    Saves a FAISS index to disk.

    Args:
        index (faiss.Index): The FAISS index to save.
        index_path (str): Path to save the index.
    """
    print(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    print("FAISS index successfully saved!")

def load_faiss_index(index_path):
    """
    Loads a FAISS index from disk.

    Args:
        index_path (str): Path to the FAISS index file.

    Returns:
        faiss.Index: The loaded FAISS index.
    """
    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded. Total vectors: {index.ntotal}")
    return index

def add_embeddings_to_index(index, new_embeddings):
    """
    Adds new embeddings to an existing FAISS index.

    Args:
        index (faiss.Index): The FAISS index to add embeddings to.
        new_embeddings (np.ndarray): Numpy array of new embeddings.
    """
    print("Adding new embeddings to the FAISS index...")
    index.add(new_embeddings)
    print(f"Total vectors in the index after addition: {index.ntotal}")

def search_similar(index, query_embedding, top_k=5):
    """
    Searches for the top_k most similar vectors in the index to the query_embedding.

    Args:
        index (faiss.Index): The FAISS index to search.
        query_embedding (np.ndarray): Numpy array of shape (dimension,) representing the query.
        top_k (int): Number of top similar vectors to retrieve.

    Returns:
        distances (np.ndarray): Distances of the top_k similar vectors.
        indices (np.ndarray): Indices of the top_k similar vectors in the index.
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    distances, indices = index.search(query_embedding, top_k)
    return distances, indices

def main():

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    # generate embeds
    embeddings_dict, metadata = generate_embeddings(INPUT_CSV, TEXT_COLUMNS, MODEL_NAME)

    # save embeddings
    save_embeddings(embeddings_dict, EMBEDDINGS_PKL)

    # save metadata
    save_metadata(metadata, METADATA_PKL)


    for col in TEXT_COLUMNS:
        print(f"\nProcessing FAISS index for column: '{col}'")
        embeddings = embeddings_dict[col]

        # create a faiss index
        index = create_faiss_index(embeddings, INDEX_TYPE)

        # save faiss index
        faiss_index_path = os.path.join(FAISS_DIR, f'faiss_index_{col}.idx')
        save_faiss_index(index, faiss_index_path)

        # load FAISS index
        loaded_index = load_faiss_index(faiss_index_path)

        # perform a similarity search

        query_embedding = embeddings[0]
        distances, indices_found = search_similar(loaded_index, query_embedding, top_k=TOP_K)

        print(f"\nTop {TOP_K} similar vectors to the first entry in column '{col}':")
        for rank, (dist, idx) in enumerate(zip(distances[0], indices_found[0]), start=1):
            print(f"Rank {rank}: Index {idx}, Distance {dist}")
            # we can retireve the original data using the metadata
            original_record = metadata[idx]
            print(f"Original Record ID: {original_record.get('id', 'N/A')}, Title: {original_record.get('title', 'N/A')}\n")

if __name__ == "__main__":
    main()
