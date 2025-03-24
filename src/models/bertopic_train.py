from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


import os

# Define the stopwords path manually (adjust path as needed)
venv_path = os.path.join(os.getcwd(), "src", "dataset", "vietnamese-stopwords.txt")

# Read stopwords with error handling
try:
    with open(venv_path, "r", encoding="utf-8") as f:
        vietnamese_stopwords = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    # Fallback to basic Vietnamese stopwords if file not found
    vietnamese_stopwords = ["và", "của", "có", "cho", "các", "là", "không",
                           "được", "trong", "đã", "với", "những", "này", "để", "từ"]
    print("Warning: Stopwords file not found. Using basic stopwords list.")


def optimize_bertopic(docs, pretrained_model):
    docs = docs.tolist()
    pretrained_model = pretrained_model

    # Precompute embeddings
    embedding_model = SentenceTransformer(pretrained_model)
    embeddings = embedding_model.encode(docs, show_progress_bar=True, normalize_embeddings=True)

    # Optimized UMAP (Better Global Structure)
    umap_model = UMAP(
        n_neighbors=15,  # Lower for Vietnamese text clustering
        n_components=5,
        min_dist=0.05,  # Tighter clusters for Vietnamese
        metric="cosine",
        random_state=42
    )

    # Optimized HDBSCAN (Smaller Topics & Fine-tuning)
    hdbscan_model = HDBSCAN(
        min_cluster_size=20,  # Lower threshold for Vietnamese topics
        min_samples=3,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    # Optimized Vectorizer (Better Text Representation)
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 3),  # Include trigrams for compound Vietnamese phrases
        min_df=3,  # Lower threshold for Vietnamese
        max_df=0.9,
        max_features=15_000,  # Increase for Vietnamese vocabulary coverage
        stop_words=vietnamese_stopwords
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,

        # Vietnamese optimization
        top_n_words=15,  # More words for better Vietnamese topic interpretation
        calculate_probabilities=True,
        verbose=True,
        nr_topics="auto",  # Let the model determine optimal number of topics
    )


    topics, probs = topic_model.fit_transform(docs, embeddings)

    return topic_model, topics, probs
