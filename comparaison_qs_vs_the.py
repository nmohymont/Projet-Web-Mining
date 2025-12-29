import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from text_mining_dimensionality_test import extract_tokens, load_parquet_data, tdm_creation, filter_matrix, tfidf_calculation

def tfidf_to_binary(tfidf_matrix, threshold=0.0):
    """
    Convert a TF-IDF matrix into a binary matrix (term presence/absence).
    
    Args:
        tfidf_matrix: TF-IDF DataFrame (rows = documents, columns = terms)
        threshold: Threshold above which a term is considered "present" (default: 0.0)
    
    Returns:
        Binary DataFrame (1 if term present, 0 otherwise)
    """
    print(f"🔄 Converting TF-IDF → Binary matrix (threshold = {threshold})...")
    
    # Convert to binary: 1 if value > threshold, 0 otherwise
    binary_matrix = (tfidf_matrix > threshold).astype(int)
    
    print(f"   Binary matrix created: {binary_matrix.shape[0]} documents × {binary_matrix.shape[1]} terms")
    print(f"   Sparsity: {100 * (1 - binary_matrix.sum().sum() / (binary_matrix.shape[0] * binary_matrix.shape[1])):.2f}%")
    
    return binary_matrix


def load_tfidf_embeddings(filename, number_uni=None):

    """
    Process university corpus: tokenization, TDM creation, filtering, TF-IDF, and BERT embeddings.
    
    Args:
        filename: Path to parquet file
        number_uni: Number of universities to process (None = all)
    
    Returns:
        Tuple (tfidf_matrix, bert_embeddings)
    """
    docs_simple, docs_stem, docs_lemma, raw_texts = load_parquet_data(filename, number_uni=number_uni)
    td_matrix = tdm_creation(docs_lemma)
    filtered = filter_matrix(td_matrix)
    tfidf = tfidf_calculation(filtered)
    
    doc_names = list(raw_texts.keys())
    documents = list(raw_texts.values())
    embeddings = model.encode(documents, normalize_embeddings=True)
    
    return tfidf, embeddings

def calculate_matched_similarity(df_mapping, tfidf_qs, tfidf_the, embeddings_qs, embeddings_the, binary_qs, binary_the):
    """
    Calculate similarity measures (TF-IDF+Cosine, BERT+Cosine, Jaccard) between matched universities.
    
    Args:
        df_mapping: DataFrame with QS_Name and THE_Name columns
        tfidf_qs: TF-IDF matrix for QS
        tfidf_the: TF-IDF matrix for THE
        embeddings_qs: BERT embeddings for QS
        embeddings_the: BERT embeddings for THE
        binary_qs: Binary matrix for QS
        binary_the: Binary matrix for THE
    
    Returns:
        DataFrame with columns: QS_Name, THE_Name, TF-IDF_Similarity, BERT_Similarity, Jaccard_Similarity
    """
    results = []
    
    for _, row in df_mapping.iterrows():
        qs_name = row['QS_Name']
        the_name = row['THE_Name']
        
        # --- TF-IDF & COSINE ---
        
        # Check if both universities exist in the matrices
        if qs_name not in tfidf_qs.index or the_name not in tfidf_the.index:
            continue
        
        # Align vocabularies (union of columns)
        all_terms = set(tfidf_qs.columns).union(set(tfidf_the.columns))
        
        # Create aligned vectors with zeros for missing terms
        vec_qs_tfidf = np.array([tfidf_qs.loc[qs_name].get(term, 0) for term in all_terms])
        vec_the_tfidf = np.array([tfidf_the.loc[the_name].get(term, 0) for term in all_terms])
        
        # Calculate TF-IDF cosine similarity
        tfidf_sim = cosine_similarity(
            vec_qs_tfidf.reshape(1, -1), 
            vec_the_tfidf.reshape(1, -1)
        )[0, 0]
        
        # --- BERT & COSINE ---
        # Get university indices in embeddings
        qs_idx = list(tfidf_qs.index).index(qs_name)
        the_idx = list(tfidf_the.index).index(the_name)
        
        # Calculate BERT cosine similarity
        bert_sim = cosine_similarity(
            embeddings_qs[qs_idx].reshape(1, -1),
            embeddings_the[the_idx].reshape(1, -1)
        )[0, 0]
        
        # --- JACCARD ---
        
        # Check if both universities exist in binary matrices
        if qs_name not in binary_qs.index or the_name not in binary_the.index:
            continue
        
        # Align vocabularies (union of columns)
        all_terms_binary = set(binary_qs.columns).union(set(binary_the.columns))
        
        # Create aligned binary vectors
        vec_qs_binary = np.array([binary_qs.loc[qs_name].get(term, 0) for term in all_terms_binary])
        vec_the_binary = np.array([binary_the.loc[the_name].get(term, 0) for term in all_terms_binary])
        
        # Calculate intersection and union
        intersection = np.sum(vec_qs_binary & vec_the_binary)  # Terms present in BOTH
        union = np.sum(vec_qs_binary | vec_the_binary)         # Terms present in AT LEAST ONE
        
        # Jaccard similarity
        if union > 0:
            jaccard_sim = intersection / union
        else:
            jaccard_sim = 0.0  # If no term is present in both documents
        
        results.append({
            'QS_Name': qs_name,
            'THE_Name': the_name,
            'TF-IDF_Similarity': tfidf_sim,
            'BERT_Similarity': bert_sim,
            'Jaccard_Similarity': jaccard_sim,
            'Intersection': intersection,
            'Union': union,
            'Score': row['Score']
        })
    
    return pd.DataFrame(results)

# Note: Uncomment the lines below during the first execution
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # Extension for lemmatization to match studies, studying and studied to a single token study
nltk.download('omw-1.4') # Additional extension for better lemmatization


# --- STEMMING CONFIGURATION ---
stemmer = nltk.stem.SnowballStemmer("english")

# --- LEMMATIZATION CONFIGURATION ---
lemmatizer = nltk.stem.WordNetLemmatizer()

stop_words = list(set(stopwords.words('english'))) + ["'s"]

# Load BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load mapping file
df_mapping = pd.read_csv("DATA/CLEAN/CSV/university_mapping_qs_the.csv")
df_mapping = df_mapping[df_mapping['Score'] > 0.857]

qs_names_to_keep = set(df_mapping['QS_Name'].dropna())
the_names_to_keep = set(df_mapping['THE_Name'].dropna())

df_qs = pd.read_parquet("DATA/CLEAN/PARQUET/qs_university_corpus.parquet")  
df_the = pd.read_parquet("DATA/CLEAN/PARQUET/the_university_corpus.parquet")

df_qs_filtered = df_qs[df_qs['name'].isin(qs_names_to_keep)]
df_the_filtered = df_the[df_the['name'].isin(the_names_to_keep)]

df_qs_filtered.to_parquet("DATA/CLEAN/PARQUET/QS_vs_THE/qs_university_corpus_matched.parquet", index=False)
df_the_filtered.to_parquet("DATA/CLEAN/PARQUET/QS_vs_THE/the_university_corpus_matched.parquet", index=False)

# Process QS corpus
tfidf_qs, embeddings_qs = load_tfidf_embeddings("DATA/CLEAN/PARQUET/QS_vs_THE/qs_university_corpus_matched.parquet")

# Process THE corpus
tfidf_the, embeddings_the = load_tfidf_embeddings("DATA/CLEAN/PARQUET/QS_vs_THE/the_university_corpus_matched.parquet")

# Convert TF-IDF matrices to binary matrices
binary_qs = tfidf_to_binary(tfidf_qs)
binary_the = tfidf_to_binary(tfidf_the)

# Calculate all similarities
similarity_results = calculate_matched_similarity(
    df_mapping, 
    tfidf_qs, 
    tfidf_the, 
    embeddings_qs, 
    embeddings_the,
    binary_qs,
    binary_the
)

# Display results
print(similarity_results.shape)

# Descriptive statistics
print("\n=== Similarity Statistics ===")
print(similarity_results[['TF-IDF_Similarity', 'BERT_Similarity', 'Jaccard_Similarity']].describe())

# Save results
# similarity_results.to_csv("QS_vs_THE/similarity_results.csv", index=False)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# TF-IDF histogram
axes[0].hist(similarity_results['TF-IDF_Similarity'], bins=20, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('TF-IDF Similarity')
axes[0].set_ylabel('Frequency')
axes[0].set_title('TF-IDF Similarity Distribution (QS vs THE)')
axes[0].axvline(similarity_results['TF-IDF_Similarity'].mean(), color='red', linestyle='--', label='Mean')
axes[0].legend()

# Jaccard histogram
axes[1].hist(similarity_results['Jaccard_Similarity'], bins=20, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_xlabel('Jaccard Similarity')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Jaccard Similarity Distribution (QS vs THE)')
axes[1].axvline(similarity_results['Jaccard_Similarity'].mean(), color='red', linestyle='--', label='Mean')
axes[1].legend()

# BERT histogram
axes[2].hist(similarity_results['BERT_Similarity'], bins=20, edgecolor='black', alpha=0.7, color='red')
axes[2].set_xlabel('BERT Similarity')
axes[2].set_ylabel('Frequency')
axes[2].set_title('BERT Similarity Distribution (QS vs THE)')
axes[2].axvline(similarity_results['BERT_Similarity'].mean(), color='red', linestyle='--', label='Mean')
axes[2].legend()

plt.tight_layout()
# plt.savefig("QS_vs_THE/similarity_distribution.png", dpi=300)
plt.show()

# Identify universities with low similarity (strong editorial divergences)
low_similarity = similarity_results[similarity_results['TF-IDF_Similarity'] < 0.4].sort_values('TF-IDF_Similarity')
print("\n=== Universities with Strong Editorial Divergence (TF-IDF < 0.4) ===")
print(low_similarity[['QS_Name', 'THE_Name', 'TF-IDF_Similarity', 'BERT_Similarity']])

# Identify universities with high similarity (coherent descriptions)
high_similarity = similarity_results[similarity_results['TF-IDF_Similarity'] > 0.7].sort_values('TF-IDF_Similarity', ascending=False)
print("\n=== Universities with Strong Coherence (TF-IDF > 0.7) ===")
print(high_similarity[['QS_Name', 'THE_Name', 'TF-IDF_Similarity', 'BERT_Similarity']])
