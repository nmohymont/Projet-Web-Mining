import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer # For BERT
import json
import os

# --- 1. CONFIGURATION AND TOOLS ---

# Note: Uncomment the lines below during the first execution
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


# Global Variables 
#------------------
# Minimum character length to keep a word
min_word_length = 3
# Matrix extract size for console display
display_cols = 10
display_rows = 10
# Number of universities to process (Set to None to process all)
num_universities = None

# --- 1. FILE PATH DEFINITIONS (INPUT) ---
file_qs = 'DATA/CLEAN/PARQUET/qs_university_corpus.parquet'
file_the = 'DATA/CLEAN/PARQUET/the_university_corpus.parquet'       # THE 2025
file_the_2012 = 'DATA/CLEAN/PARQUET/the_university_corpus_2011-2012.parquet'
file_the_2021 = 'DATA/CLEAN/PARQUET/the_university_corpus_2021.parquet'

# --- 2. SELECT FILE TO PROCESS (CURRENT_MODE) ---
# Available options: 'qs', 'the', 'the_2012', 'the_2021'
CURRENT_MODE = 'the' 

# Directory for JSON output
output_dir = os.path.join('DATA', 'CLEAN', 'JSON')

# Selection logic for input file and output filename
if CURRENT_MODE == 'qs':
    input_file = file_qs
    json_filename = 'university_processed_features_qs.json'
    print(f"--> ACTIVE MODE: QS 2025")

elif CURRENT_MODE == 'the':
    input_file = file_the
    json_filename = 'university_processed_features_the.json'
    print(f"--> ACTIVE MODE: THE 2025")

elif CURRENT_MODE == 'the_2012':
    input_file = file_the_2012
    json_filename = 'university_processed_features_the_2012.json'
    print(f"--> ACTIVE MODE: THE 2011-2012")

elif CURRENT_MODE == 'the_2021':
    input_file = file_the_2021
    json_filename = 'university_processed_features_the_2021.json'
    print(f"--> ACTIVE MODE: THE 2021")

else:
    raise ValueError("Unknown CURRENT_MODE.")

output_file_path = os.path.join(output_dir, json_filename)


# --- STEMMING CONFIGURATION ---
stemmer = nltk.stem.SnowballStemmer("english")

# --- LEMMATIZATION CONFIGURATION ---
lemmatizer = nltk.stem.WordNetLemmatizer()


# --- CUSTOM BLACKLIST (DOMAIN SPECIFIC STOP WORDS) ---
BLACKLIST = [
    'qs', 'http', 'https', 'www', 'com', 'org', 
    'one', 'two', 'also', 'since', 'many', 'well'
]

# --- STOP WORDS CONFIGURATION ---
stop_words = list(set(stopwords.words('english'))) + ["'s"] + BLACKLIST

def extract_tokens(text, mode='stem'):
    text = str(text).lower() # Lowercase 

    # Remove English contractions
    text = text.replace("’", "'")  
    text = re.sub(r"'\w+", '', text)

    tokens = nltk.word_tokenize(text) 
    processed_tokens = []
    
    for t in tokens:
        # filter: alphabetic, not punctuation, not a stopword, and meets length requirement
        if t.isalpha() and t not in string.punctuation and t not in stop_words and len(t) >= min_word_length: 
            if mode == 'stem':
                processed_tokens.append(stemmer.stem(t))
            elif mode == 'lemma':
                processed_tokens.append(lemmatizer.lemmatize(t))
            elif mode == 'none':  
                processed_tokens.append(t)
                
    return processed_tokens


def load_parquet_data(path, limit=None): 
    df = pd.read_parquet(path) 
    
    if limit is not None:
        df = df.iloc[:limit]
    
    docs_none = {}    # Tokenization only
    docs_stem = {}      
    docs_lemma = {}     
    raw_texts = {} 
    
    for _, row in df.iterrows(): 
        name = row['name']
        desc = row['description']

        docs_none[name] = extract_tokens(desc, mode='none') 
        docs_stem[name] = extract_tokens(desc, mode='stem')
        docs_lemma[name] = extract_tokens(desc, mode='lemma')
        raw_texts[name] = desc 
        
    return docs_none, docs_stem, docs_lemma, raw_texts

def test_threshold(matrix_full):
    # Prepare data for the elbow plot
    doc_counts = (matrix_full > 0).sum(axis=0).sort_values(ascending=False)
    doc_counts_pct = (doc_counts / len(matrix_full)) * 100

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(doc_counts_pct)), doc_counts_pct, color='teal', linewidth=2)

    # Draw cutoff zones
    plt.axhline(y=2, color='r', linestyle='--', label='min_df threshold 2%')
    plt.axhline(y=85, color='orange', linestyle='--', label='max_df threshold 85%')

    plt.title("Elbow Method Visualization for DF Thresholds")
    plt.xlabel("Number of Unique Words (Ordered by Frequency)")
    plt.ylabel("Presence in Documents (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Apply filter after visual analysis
    matrix_filtered = filter_matrix(matrix_full, min_df_pct=0.02, max_df_pct=0.85)

    print(f"Words before filtering: {matrix_full.shape[1]}")
    print(f"Words after filtering: {matrix_filtered.shape[1]}")

    # Test thresholds
    test_range = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    results = []

    for s in test_range:
        filtered = filter_matrix(matrix_full, min_df_pct=s, max_df_pct=0.75)
        sparsity = calculate_sparsity(filtered)
        n_words = filtered.shape[1]
        results.append({'threshold': s*100, 'sparsity': sparsity, 'words': n_words})

    df_res = pd.DataFrame(results)
    print("Sparsity Analysis Results:")
    print(df_res)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('min_df Threshold (%)')
    ax1.set_ylabel('Sparsity (%)', color='tab:red')
    ax1.plot(df_res['threshold'], df_res['sparsity'], marker='o', color='tab:red', label='Sparsity')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of Remaining Words', color='tab:blue')
    ax2.bar(df_res['threshold'], df_res['words'], alpha=0.3, color='tab:blue', label='Words')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title("Impact of min_df Threshold on Sparsity and Vocabulary Size")
    plt.show()


# Term-document Matrix creation
def create_tdm(docs_dict):
    vocab = set(token for tokens in docs_dict.values() for token in tokens)
    term_freqs = {doc: Counter(tokens) for doc, tokens in docs_dict.items()}
    matrix = pd.DataFrame(
        {term: [term_freqs[doc].get(term, 0) for doc in docs_dict] for term in vocab},
        index=docs_dict.keys()
    )
    return matrix

def filter_matrix(matrix, min_df_pct=0.02, max_df_pct=0.75):
    n_docs = len(matrix)
    doc_freq = (matrix > 0).sum(axis=0)
    
    min_counts = n_docs * min_df_pct
    max_counts = n_docs * max_df_pct  

    mask = (doc_freq >= min_counts) & (doc_freq <= max_counts)
    return matrix.loc[:, mask]

def calculate_sparsity(matrix):
    total_cells = matrix.size
    zero_cells = (matrix == 0).astype(int).sum().sum()
    return (zero_cells / total_cells) * 100

def calculate_tfidf(filtered_matrix):
    row_sums = filtered_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1 
    tf = filtered_matrix.div(row_sums, axis=0)
    df_count = (filtered_matrix > 0).sum(axis=0)
    N = filtered_matrix.shape[0]
    idf = np.log(N / df_count)
    return tf.mul(idf, axis=1)

def analyze_and_justify(stem_cos, lemma_cos, bert_cos):
    
    def get_stats(df, name):
        # Extract upper triangle to avoid duplicates and self-similarity diagonal
        vals = df.values[np.triu_indices_from(df, k=1)]
        return {
            "Method": name, 
            "Mean": np.mean(vals), 
            "Median": np.median(vals), 
            "_vals": vals, 
            "Max" : np.max(vals), 
            "Min": np.min(vals)
        }

    # Compare 3 candidates (Cosine similarity)
    candidates = [
        get_stats(stem_cos, "Stem (Cos)"),
        get_stats(lemma_cos, "Lemma (Cos)"),
        get_stats(bert_cos, "BERT (Cos)")
    ]
    
    # 1. Statistical Table
    df_res = pd.DataFrame([{k: v for k, v in d.items() if k != '_vals'} for d in candidates])
    print("\n--- Similarity Score Table (Mean values) ---")
    print(df_res.sort_values("Mean", ascending=False).to_string(index=False))
    
    # 2. Boxplot Visualization
    plt.figure(figsize=(10, 6))
    colors = ['lightblue', 'lightgreen', 'orange']
    
    bplot = plt.boxplot([d['_vals'] for d in candidates], labels=[d['Method'] for d in candidates], patch_artist=True)
    
    for patch, color in zip(bplot['boxes'], colors): 
        patch.set_facecolor(color)
        
    plt.title("Similarity Comparison (Cosine): Stem vs Lemma vs BERT", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylabel("Similarity Score (0-1)")
    plt.show()

    # 3. FINAL VERDICT
    print("\n=== VERDICT ===")
    
    winner = df_res.loc[df_res['Mean'].idxmax()]
    print(f"   Method with highest mean: {winner['Method']} ({winner['Mean']:.4f})")
    
    if "BERT" in winner['Method']:
        print("   BERT offers the highest scores, indicating strong semantic coherence detected.")
    else:
        print("   A classic method (Stem/Lemma) provides higher scores than BERT.")
        print("   This suggests significant vocabulary repetition (exact word matching),")
        print("   where BERT might be stricter or more nuanced regarding context.")


# --- 2. FULL PIPELINE EXECUTION ---

print("=== STEP 1: RAW TEXT RETRIEVAL ===")
docs_none, docs_stem, docs_lemma, raw_texts = load_parquet_data(input_file, num_universities)

first_uni = list(docs_stem.keys())[0]
print(f"University: {first_uni} (Data loaded)\n")


print(" ")
print("=== STEP 2: TOKENIZATION LEVELS COMPARISON ===")

print(f"A) SIMPLE Tokens:")
print(f"   {docs_none[first_uni][:20]}...") 

print(f"\nB) LEMMATIZED Tokens:")
print(f"   {docs_lemma[first_uni][:20]}...")

print(f"\nC) STEMMED Tokens:")
print(f"   {docs_stem[first_uni][:20]}...")


# --- PARALLEL PIPELINE ---
print("---------------------------------------------------")
print("=== STEP 3: INITIAL MATRICES ===")

# Create matrices
tdm_none = create_tdm(docs_none) 
tdm_stem = create_tdm(docs_stem)
tdm_lemma = create_tdm(docs_lemma)

print("1. NONE Matrix (Dimensions):", tdm_none.shape)
print("2. LEMMA Matrix (Dimensions):", tdm_lemma.shape)
print("3. STEM Matrix (Dimensions):", tdm_stem.shape)
print("-" * 40)


print("\n=== STEP 4: FILTERED MATRICES & THRESHOLDING ===")

# Apply filters
filtered_stem = filter_matrix(tdm_stem)
filtered_lemma = filter_matrix(tdm_lemma)

print(f"Dimensions after filter (Stem): {filtered_stem.shape}")
print(f"Dimensions after filter (Lemma): {filtered_lemma.shape}")

# Optional: Threshold visual analysis (Uncomment to run)
# test_threshold(tdm_lemma)

print("\n=== STEP 5: TF-IDF CALCULATION ===")
tfidf_stem = calculate_tfidf(filtered_stem)
tfidf_lemma = calculate_tfidf(filtered_lemma)

print("TF-IDF Computation Complete.")


print("\n" + "="*60)
print("TF-IDF MATRIX PREVIEW (Top 10 Universities x Top 10 Tokens)")
print("="*60)

# Configure pandas to display all columns in the prompt without wrapping
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 4) # Adjust precision for better readability

# Display an extract using the display_rows and display_cols variables
print(tfidf_lemma.iloc[:display_rows, :display_cols])
print("="*60 + "\n")

print("\n=== STEP 6: BERT VECTORIZATION ===")
doc_names = list(raw_texts.keys())
documents = list(raw_texts.values())

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding documents with BERT...")
embeddings = model.encode(documents, normalize_embeddings=True)

# Similarity matrix for BERT
similarity_matrix_bert = cosine_similarity(embeddings)
similarity_df_bert = pd.DataFrame(similarity_matrix_bert, index=doc_names, columns=doc_names)


print("\n=== STEP 7: SIMILARITY CALCULATION (Cosine Comparison) ===")

# 1. Stemming (Cosine)
print("Computing Cosine Similarity for Stemming...")
sim_stem_cos = pd.DataFrame(cosine_similarity(tfidf_stem), index=doc_names, columns=doc_names)

# 2. Lemmatization (Cosine)
print("Computing Cosine Similarity for Lemmatization...")
sim_lemma_cos = pd.DataFrame(cosine_similarity(tfidf_lemma), index=doc_names, columns=doc_names)

# 3. BERT (Cosine) 
print("Formatting BERT Similarity Data...")
sim_bert_cos = similarity_df_bert

# Run comparative analysis
analyze_and_justify(sim_stem_cos, sim_lemma_cos, sim_bert_cos)


# --- 8. FINAL FILTERING AND STORAGE ---

print("\n=== STEP 8: SAVE DATA (JSON Export - Top TF-IDF) ===")

# Mapping Country -> Region
country_to_region = {
    "United Kingdom": "Europe", "Switzerland": "Europe", "Germany": "Europe", "Sweden": "Europe",
    "Belgium": "Europe", "France": "Europe", "Netherlands": "Europe", "Denmark": "Europe",
    "Finland": "Europe", "Norway": "Europe", "Spain": "Europe", "Ireland": "Europe",
    "Austria": "Europe", "Italy": "Europe", "Russian Federation": "Europe", "Czechia": "Europe",
    "Greece": "Europe", "Portugal": "Europe", "Cyprus": "Europe", "Poland": "Europe",
    "Iceland": "Europe", "Hungary": "Europe", "Luxembourg": "Europe", "Estonia": "Europe",
    "Romania": "Europe", "Slovenia": "Europe", "Malta": "Europe", "Lithuania": "Europe",
    "Latvia": "Europe", "Ukraine": "Europe", "Serbia": "Europe", "Slovakia": "Europe",
    "Croatia": "Europe", "Bulgaria": "Europe", "North Macedonia": "Europe",
    "Bosnia and Herzegovina": "Europe", "Kosovo": "Europe",

    "United States": "North America", "Canada": "North America", "Mexico": "North America",
    "Costa Rica": "North America", "Jamaica": "North America", "Cuba": "North America",

    "China": "Asia", "Singapore": "Asia", "Japan": "Asia", "Hong Kong": "Asia",
    "South Korea": "Asia", "Taiwan": "Asia", "Macao": "Asia", "Saudi Arabia": "Asia",
    "United Arab Emirates": "Asia", "India": "Asia", "Qatar": "Asia", "Malaysia": "Asia",
    "Lebanon": "Asia", "Turkey": "Asia", "Iran": "Asia", "Brunei Darussalam": "Asia",
    "Jordan": "Asia", "Bahrain": "Asia", "Kazakhstan": "Asia", "Pakistan": "Asia",
    "Thailand": "Asia", "Uzbekistan": "Asia", "Vietnam": "Asia", "Kuwait": "Asia",
    "Bangladesh": "Asia", "Indonesia": "Asia", "Philippines": "Asia", "Sri Lanka": "Asia",
    "Fiji": "Asia", "Georgia": "Asia", "Armenia": "Asia",
}

# Keep all filtered tokens
top_k_val = len(tfidf_lemma.columns)  

# Token Selection
final_lemma_tokens = {}
print("Selecting top-k signature tokens for each university...")
for univ_name, row in tfidf_lemma.iterrows():
    top_tokens = row.nlargest(top_k_val)
    valid_tokens = top_tokens[top_tokens > 0].index.tolist()
    final_lemma_tokens[univ_name] = valid_tokens

# Process Metadata
df_meta = pd.read_parquet(input_file)
df_meta["name"] = df_meta["name"].astype(str).str.strip()

# Apply Region Logic
if CURRENT_MODE.lower() in ["the", "the_2021"]:
    df_meta["country"] = df_meta["country"].astype(str).str.strip()
    df_meta["region"] = df_meta["country"].map(country_to_region).fillna("Other")
else:
    if "region" in df_meta.columns:
        df_meta["region"] = df_meta["region"].astype(str).str.strip().replace("", "Other")
    else:
        df_meta["region"] = "Other"

# Rank Sanitization
def clean_rank(val):
    try:
        if pd.isna(val) or str(val).strip() == "": return None
        val_str = str(val).strip().replace('=', '').replace('–', '-')
        num_part = val_str.split('-')[0].split('+')[0].strip()
        return float(num_part)
    except:
        return None

rank_column_name = "rank" if "rank" in df_meta.columns else "ranking"
if rank_column_name in df_meta.columns:
    df_meta["rank_cleaned"] = df_meta[rank_column_name].apply(clean_rank)
else:
    df_meta["rank_cleaned"] = None

# Build final dictionaries
regions_final = df_meta.set_index("name")["region"].reindex(final_lemma_tokens.keys()).fillna("Other").to_dict()
ranks_final = df_meta.set_index("name")["rank_cleaned"].reindex(final_lemma_tokens.keys()).to_dict()

# JSON Structure assembly
json_output_data = {
    "info": {
        "mode": CURRENT_MODE,
        "num_universities": len(final_lemma_tokens),
        "total_vocab_size": tfidf_lemma.shape[1],
        "top_k_per_university": top_k_val
    },
    "tokens": final_lemma_tokens,
    "regions": regions_final,
    "ranks": ranks_final 
}

# Custom Encoder for Numpy Types
class NumpyDataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyDataEncoder, self).default(obj)

print(f"Saving results to: {output_file_path} ...")
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(json_output_data, f, cls=NumpyDataEncoder, ensure_ascii=False, indent=4)

print("Success! JSON file generated.")