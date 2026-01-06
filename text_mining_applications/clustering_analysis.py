import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

from nltk.tokenize import word_tokenize # tokenization based on word
from nltk.stem import SnowballStemmer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from collections import Counter
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

# StandardScaler scales features to zero mean and unit variance (column-wise feature scaling).

# Don't forget to uncomment this dowload needed during the first time running this code
# Download the nltk
#nltk.download('punkt')
#nltk.download('snowball_data')

stemmer = SnowballStemmer("english")

def filter_description_window_stemmed(row, token_dict_stemmed, window_size=5):
    """ 
    1. Use Stemming to match words (e.g., 'studying' matches 'studi').
    2. Keep a window of words around the match (not the whole sentence).
    """
    univ_key = row['name'] 
    description = str(row['description'])
    
    # Retrieve root stems (already stemmed in main)
    target_stems = token_dict_stemmed.get(univ_key)
    
    if target_stems is None:
        return "MISSING_KEY"
    if not description:
        return ""
    
    # Word tokenization breaks the description into a list of words in the order of the text
    words = word_tokenize(description)
    
    indices_to_keep = set()
    
    # Iterate word by word
    for i, w in enumerate(words):
        # i is the index of the token
        # w is the word

        # Stem the current word to see if it matches a target token
        w_stem = stemmer.stem(w.lower())
        
        # Match is done on the stem!
        if w_stem in target_stems:

            # If matched, mark the window around it
            start = max(0, i - window_size) 
            # use max function if the w_stem match the target and is at the begining of the list of words
            end = min(len(words), i + window_size + 1) # +1 because end in the range (start,end) is excluded
            # use min if the target is at the end of the list of words
            for idx in range(start, end):
                indices_to_keep.add(idx)
    
    # Reconstruction
    if not indices_to_keep:
        return ""
    
    sorted_indices = sorted(list(indices_to_keep))
    kept_words = []
    
    last_idx = -1
    
    # text reconstruction by stiching only the selected windows
    # inserting "..." wherever there is a gap between retained word indices
    for idx in sorted_indices:
        if last_idx != -1 and idx > last_idx + 1:
            kept_words.append("...") 
        kept_words.append(words[idx])
        last_idx = idx
        
    return " ".join(kept_words)

input_desc = "DATA/CLEAN//PARQUET/qs_university_corpus.parquet"
input_file = "DATA/CLEAN/JSON/top_100_token_qs.json"

df = pd.read_parquet(input_desc)
df = df.dropna(subset=['description']) # Safety drop
df['description'] = df['description'].astype(str).str.strip()

with open(input_file, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

univ_to_tokens_stemmed = {}
raw_univ_to_tokens = full_data["tokens"]

for univ, tokens in raw_univ_to_tokens.items():
    # Store stems to match broader variations (study == studies)
    stemmed_tokens = set([stemmer.stem(t) for t in tokens])
    univ_to_tokens_stemmed[univ] = stemmed_tokens

print("Processing descriptions filtering...")

df['filtered_desc'] = df.apply(
    lambda row: filter_description_window_stemmed(row, univ_to_tokens_stemmed, window_size=5), axis=1
)
#lambda row used to select rows one at a time
#axis = 1 to work on columns

# --- 4. RESULTS ANALYSIS ---
missing_keys = df[df['filtered_desc'] == "MISSING_KEY"].shape[0]
empty_desc = df[df['filtered_desc'] == ""].shape[0]
success = df[~df['filtered_desc'].isin(["MISSING_KEY", ""])].shape[0]

print(f"\n--- RESULTS ---")
print(f"Universities not found in JSON (MISSING_KEY): {missing_keys}")
print(f"Empty descriptions after filtering (No keywords found): {empty_desc}")
print(f"Successfully optimized descriptions: {success}")

if missing_keys > 0:
    print("\nExample of non-matching names:")
    print(df[df['filtered_desc'] == "MISSING_KEY"]['name'].head(3).tolist())

# --- 5. EXPORT FOR BERT ---
# Keep only successful matches
df_final = df[~df['filtered_desc'].isin(["MISSING_KEY", ""])].copy()
texts_optimized = df_final['filtered_desc'].tolist() # take a column of the dataframe to a list of strings because BERT needs sentence or parts of sentence
print(f"\n'texts_optimized' list ready with {len(texts_optimized)} elements for BERT.")

# --- 6. BERT PREPARATION ---
# Keep only rows with non-empty filtered text
df_final = df[df['filtered_desc'] != ""].copy()
texts_optimized = df_final['filtered_desc'].tolist()
univ_names_final = df_final['name'].tolist()

print(f"Ready for BERT with {len(texts_optimized)} optimized descriptions.")

# === 2. BERT EMBEDDINGS ON DESCRIPTIONS ===
model_name = 'sentence-transformers/all-mpnet-base-v2' 
embed_model = SentenceTransformer(model_name)

embeddings = embed_model.encode(texts_optimized, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
#np.asarray is a safety check to make sure the embeddings is a NumPy table
embeddings = np.asarray(embeddings)
print("Embeddings shape:", embeddings.shape)

# Standardization: crucial for PCA
# Ensures each BERT dimension has the same statistical weight
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# PCA Application
pca_components = 30
pca = PCA(n_components=pca_components, random_state=42)
emb_reduced = pca.fit_transform(embeddings_scaled)

# Notes: With 30 components, we reach a score of 0.2072 on the TOP 100 tokens
print(f"Original dimensions: {embeddings.shape[1]}")
print(f"Dimensions after PCA: {emb_reduced.shape[1]}")

explained_variance = pca.explained_variance_ratio_.sum()
print(f"Explained variance by {pca_components} dimensions: {explained_variance:.1%}")

# === 3. HIERARCHICAL CLUSTERING + DENDROGRAM ===

max_points = 150 # Prevents too many branches in display
if len(emb_reduced) > max_points:
    idx_sample = np.random.RandomState(42).choice(len(emb_reduced), size=max_points, replace=False)
    emb_for_dendro = emb_reduced[idx_sample]
else:
    emb_for_dendro = emb_reduced

# Linkage (Ward is well suited for Euclidean embeddings)
Z = linkage(emb_for_dendro, method='ward')

print("Hierarchical Clustering Silhouette Scores:")
for k in range(2, 11):
    # "Cut" the tree of the dendrogram to get exactly k clusters
    labels = fcluster(Z, t=k, criterion='maxclust')
    
    # Calculate score
    score = silhouette_score(emb_for_dendro, labels, metric='cosine')
    print(f"k={k} | Silhouette (Hierarchical): {score:.4f}")

plt.figure(figsize=(12, 6))
dendrogram(Z, leaf_rotation=90, leaf_font_size=8, no_labels=True)
plt.title('Hierarchical Dendrogram (Ward) - Sample')
plt.tight_layout()
plt.show()

# === CLUSTERS ANALYSIS (after choosing k) ===

# Clustering on FULL dataset ---
print("\n=== Building complete linkage ===")
Z_full = linkage(emb_reduced, method='ward')

# Testing multiple k values
k_list = [5, 7, 9]

for k_optimal in k_list:
    print("\n" + "="*80)
    print(f"=== ANALYSIS | k = {k_optimal} clusters ===")
    
    # Cut the tree
    labels_final = fcluster(Z_full, t=k_optimal, criterion='maxclust')
    
    # Global score
    sil_global = silhouette_score(emb_reduced, labels_final, metric='cosine')
    print(f"Global Silhouette: {sil_global:.4f}")
    
    # Dataframe copy for this iteration
    df_k = df_final.copy()
    df_k['cluster'] = labels_final
    df_k['silhouette'] = silhouette_samples(emb_reduced, labels_final, metric='cosine')
    
    # Score per cluster
    print("\n--- Silhouette per Cluster ---")
    for cid in sorted(df_k['cluster'].unique()):
        cluster_sil = df_k[df_k['cluster'] == cid]['silhouette']
        print(f"  Cluster {cid} | Size={cluster_sil.count():3d} | Mean_Silhouette={cluster_sil.mean():.4f}")
    
    # Top tokens and region per cluster
    print("\n--- Clusters (Top 5 tokens + Dominant Region) ---")
    for cid in sorted(df_k['cluster'].unique()):
        univs = df_k[df_k['cluster'] == cid]['name'].tolist()
        
        # Top 5 tokens
        all_tokens = []
        for u in univs:
            all_tokens.extend(raw_univ_to_tokens.get(u, []))
        top_5_terms = [t for t, _ in Counter(all_tokens).most_common(5)]
        
        # Dominant region
        all_regions = []
        for u in univs:
            region = full_data["regions"].get(u, "Unknown")
            all_regions.append(region)
        
        dominant_region, region_count = Counter(all_regions).most_common(1)[0]
        
        # Compact display
        print(f"\nCluster {cid}")
        print(f"  Top tokens: {', '.join(top_5_terms)}")
        print(f"  Dominant Region: {dominant_region} ({region_count}/{len(univs)} universities)")
        print(f"  Examples: {', '.join(univs[:3])}")