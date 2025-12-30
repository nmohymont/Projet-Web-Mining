import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sentence_transformers import SentenceTransformer # For BERT
import pickle # Librairie standard pour sauvegarder des variables
import json
import os

# --- 1. CONFIGURATION AND TOOLS ---

# Note: Uncomment the lines below during the first execution
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet') #extension for the lemmatisation to match studies, studying and studied to a single token study
#nltk.download('omw-1.4') # additional extension for a better lemmatisation
#nltk.download('vader_lexicon') # for the Sentiment Intensity Analyzer


# Variables 
#------------------
# Minimum words to keep a column in the term-document matrix
number_min_words = 3
# Matrix extract size to display
matrix_size_column = 10
matrix_size_line = 10
# Number of universities to process (Set to None to process all)
number_uni = None

# --- 1. DÉFINITION DES CHEMINS DE FICHIERS (INPUT) ---
file_qs = 'DATA/CLEAN/PARQUET/qs_university_corpus.parquet'
file_the = 'DATA/CLEAN/PARQUET/the_university_corpus.parquet'       # THE 2025
file_the_2012 = 'DATA/CLEAN/PARQUET/the_university_corpus_2011-2012.parquet'
file_the_2021 = 'DATA/CLEAN/PARQUET/the_university_corpus_2021.parquet'

# --- 2. CHOIX DU FICHIER A TRAITER (CURRENT_MODE) ---
# Modifie cette variable pour lancer le traitement sur le fichier voulu.
# Options disponibles : 'qs', 'the', 'the_2012', 'the_2021'



CURRENT_MODE = 'qs'  # <--- Change ici : 'qs', 'the', 'the_2012', 'the_2021'

# Dossier où on va sauvegarder les JSON
output_dir = os.path.join('DATA', 'CLEAN', 'JSON')

# Logique de sélection du fichier d'entrée et du NOM de sortie
if CURRENT_MODE == 'qs':
    file_input = file_qs
    filename_json = 'donnees_traitees_qs.json'
    print(f"--> MODE ACTIVE : QS 2025")

elif CURRENT_MODE == 'the':
    file_input = file_the
    filename_json = 'donnees_traitees_the.json'
    print(f"--> MODE ACTIVE : THE 2025")

elif CURRENT_MODE == 'the_2012':
    file_input = file_the_2012
    filename_json = 'donnees_traitees_the_2012.json'
    print(f"--> MODE ACTIVE : THE 2011-2012")

elif CURRENT_MODE == 'the_2021':
    file_input = file_the_2021
    filename_json = 'donnees_traitees_the_2021.json'
    print(f"--> MODE ACTIVE : THE 2021")

else:
    raise ValueError("CURRENT_MODE inconnu.")

# On construit le chemin complet pour plus tard
file_output = os.path.join(output_dir, filename_json)



# --- STEMMING CONFIGURATION ---
stemmer = nltk.stem.SnowballStemmer("english")

# --- LEMMATIZATION CONFIGURATION ---
lemmatizer = nltk.stem.WordNetLemmatizer()


# --- CUSTOM BLACKLIST (DOMAIN SPECIFIC STOP WORDS) ---
BLACKLIST = [
    'qs' ,'us','uk', 'http', 'https', 'www', 'com', 'org', 
    'one', 'two', 'also', 'since', 'many', 'well'
]

# --- STOP WORDS CONFIGURATION ---
stop_words = list(set(stopwords.words('english'))) + ["'s"] + BLACKLIST

# stopwords.words('english') => list provided by NLTK library containing common English words with no significant meaning ("the", "is", "at", "which", "on", "a", "an", "and"...)
# set => Removes duplicates
# ["'s", "also", "thi", "one", "two",] => Adding context-specific words => Our own rules

def extract_tokens(text, mode='stem'):
    # ... (start of function unchanged) ...
    text = str(text).lower() # Lowercase 

    # Radical option: remove English contractions => '
    text = text.replace("’", "'")  
    text = re.sub(r"'\w+", '', text)

    tokens = nltk.word_tokenize(text) 
    processed_tokens = []
    
    for t in tokens:
        # --- MODIFICATION ICI : ajout de 'and len(t) > 2' ---
        # Cela supprime tous les mots de 1 ou 2 lettres (comme 'q', 'is', 'to')
        if t.isalpha() and t not in string.punctuation and t not in stop_words and len(t) > 2: 
            if mode == 'stem':
                processed_tokens.append(stemmer.stem(t))
            elif mode == 'lemma':
                processed_tokens.append(lemmatizer.lemmatize(t))
            elif mode == 'none':  
                processed_tokens.append(t) # Add word without modification
                
    return processed_tokens

# What was processed in "extract_tokens":

# ------Lowercase: Everything is lowercase.
# ------Tokenization: Text is split into words.
# ------Punctuation Removal: Commas, dots, etc., are gone.
# ------Stop Words Removal: Words like "the", "is", "at" are ALREADY gone. 

def load_parquet_data(filename, number_uni=None): 
    df = pd.read_parquet(filename) 
    
    #print(number_uni)
    #print(len(df))

    if number_uni is not None :
        df = df.iloc[:number_uni]
    
    docs_simple = {}    # Tokenization only
    docs_stem = {}      
    docs_lemma = {}     
    raw_texts = {} 
    
    for _, row in df.iterrows(): #iteration on the rows of the data frame
        titre = row['name']
        desc = row['description']

              
        docs_simple[titre] = extract_tokens(desc, mode='none') 
        docs_stem[titre] = extract_tokens(desc, mode='stem')
        docs_lemma[titre] = extract_tokens(desc, mode='lemma')
        raw_texts[titre] = desc 
        
    return docs_simple, docs_stem, docs_lemma, raw_texts

def test_threshold (matrix_full) :
    # 3. Prepare data for the elbow plot
    doc_counts = (matrix_full > 0).sum(axis=0).sort_values(ascending=False)
    doc_counts_pct = (doc_counts / len(matrix_full)) * 100

    # 4. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(doc_counts_pct)), doc_counts_pct, color='teal', linewidth=2)

    # Draw cutoff zones to assist decision-making
    plt.axhline(y=2, color='r', linestyle='--', label='min_df threshold 2%')
    plt.axhline(y=85, color='orange', linestyle='--', label='max_df threshold 85%')

    plt.title("Elbow Method Visualization for DF Thresholds")
    plt.xlabel("Number of Unique Words (Ordered by Frequency)")
    plt.ylabel("Presence in Documents (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # 5. Apply filter after visual analysis
    # Updated to 0.02 (2%) based on your analysis of the sparsity elbow
    matrix_filtered = filter_matrix(matrix_full, min_df_percent=0.02, max_df_percent=0.85)

    print(f"Number of words before filtering: {matrix_full.shape[1]}")
    print(f"Number of words after filtering: {matrix_filtered.shape[1]}")


    # List of min_df thresholds to test (as percentages)
    test_thresholds = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    results = []

    for s in test_thresholds:
        # Apply filter (keeping max_df constant, e.g., 0.75)
        filtered = filter_matrix(matrix_full, min_df_percent=s, max_df_percent=0.75)
        
        sparsity = calculate_sparsity(filtered)
        n_words = filtered.shape[1]
        results.append({'threshold': s*100, 'sparsity': sparsity, 'words': n_words})

    # Display results
    df_res = pd.DataFrame(results)
    print("Sparsity Analysis Results:")
    print(df_res)

    # Visualization
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


# Term-document Matrix
def tdm_creation(documents_dict):
    vocabulaire = set(token for tokens in documents_dict.values() for token in tokens)
    term_frequencies = {doc: Counter(tokens) for doc, tokens in documents_dict.items()}
    matrix = pd.DataFrame(
        {term: [term_frequencies[doc].get(term, 0) for doc in documents_dict] for term in vocabulaire},
        index=documents_dict.keys()
    )
    return matrix

def filter_matrix(matrix, min_df_percent =0.02, max_df_percent=0.75):
    number_doc = len(matrix)

    doc_freq = (matrix > 0).sum(axis=0)

    too_frequent = doc_freq[doc_freq > (number_doc * max_df_percent)].index.tolist()

    #print(f"\n Words removed because they appear in more than {max_df_percent*100}% of the documents:")
    #print(too_frequent)

    min_counts = number_doc * min_df_percent
    max_counts = number_doc * max_df_percent  

    mask = (doc_freq>= min_counts) & (doc_freq <= max_counts)

    return matrix.loc[:, mask]

def calculate_sparsity(matrix):
    # Total number of cells (rows * columns)
    total_cells = matrix.size
    # Number of zeros (values equal to 0)
    zero_cells = (matrix == 0).astype(int).sum().sum()
    # Sparsity percentage
    return (zero_cells / total_cells) * 100

def tfidf_calculation(filtered_matrix):
    row_sums = filtered_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1 
    tf = filtered_matrix.div(row_sums, axis=0)
    df_count = (filtered_matrix > 0).sum(axis=0)
    N = filtered_matrix.shape[0]
    idf = np.log(N / df_count)
    return tf.mul(idf, axis=1)

def plot_comparison(df_stem, df_lemma):
    # Creating a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Configuration of data to iterate
    data_list = [
        (axes[0], df_stem, "Similarity - STEMMING "),
        (axes[1], df_lemma, "Similarity - LEMMATIZATION ")
    ]

    for ax, df, titre in data_list:
        im = ax.imshow(df, interpolation='nearest', cmap='viridis')
        ax.set_title(titre, fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=90)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)

        for i in range(len(df)):
            for j in range(len(df)):
                val = df.iloc[i, j]
                couleur_texte = 'black' if val > 0.7 else 'white'
                ax.text(j, i, f"{val:.2f}", 
                        ha='center', va='center', color=couleur_texte, fontsize=8)

    plt.tight_layout() 
    plt.show()


# Fonction utilitaire pour convertir distance euclidienne en similarité (0-1)
def dist_to_sim(dist_matrix):
    return 1 / (1 + dist_matrix)


def analyze_6_methods(stem_cos, stem_euc, lemma_cos, lemma_euc, bert_cos, bert_euc):
    
    def get_stats(df, name):
        matrix = df.values
        # On exclut la diagonale (sim = 1.0)
        upper_indices = np.triu_indices_from(matrix, k=1)
        values = matrix[upper_indices]
        return {
            "Méthode": name,
            "Moyenne": np.mean(values),
            "Médiane": np.median(values),
            "Max": np.max(values),
            "_values": values
        }

    # Liste des 6 candidats
    candidates = [
        get_stats(stem_cos, "Stem (Cos)"),
        get_stats(stem_euc, "Stem (Euc)"),
        get_stats(lemma_cos, "Lemma (Cos)"),
        get_stats(lemma_euc, "Lemma (Euc)"),
        get_stats(bert_cos, "BERT (Cos)"),
        get_stats(bert_euc, "BERT (Euc)")
    ]
    
    # 1. Tableau
    df_res = pd.DataFrame([{k: v for k, v in d.items() if k != '_values'} for d in candidates])
    print("\n--- Tableau des Scores (Moyenne de similarité hors diagonale) ---")
    print(df_res.round(4).to_string(index=False))
    
    # 2. Graphique Boxplot
    plt.figure(figsize=(14, 7))
    
    data = [d['_values'] for d in candidates]
    labels = [d['Méthode'] for d in candidates]
    
    # Couleurs par paire : Bleu (Stem), Vert (Lemma), Rouge (BERT)
    colors = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightcoral', 'lightcoral']
    
    bplot = plt.boxplot(data, labels=labels, patch_artist=True)
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
    plt.title("Comparaison des Distributions de Similarité (6 Méthodes)", fontsize=16)
    plt.ylabel("Score de Similarité (0 = Différent, 1 = Identique)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Ajout d'une légende manuelle pour les couleurs
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='Stemming'),
        Patch(facecolor='lightgreen', label='Lemmatization'),
        Patch(facecolor='lightcoral', label='BERT')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.show()
    
    # 3. Verdict
    best = df_res.loc[df_res['Moyenne'].idxmax()]
    print(f"\n Gagnant : {best['Méthode']} avec une moyenne de {best['Moyenne']:.4f}")
    
    # Analyse rapide
    print("\n Analyse rapide :")
    print("- Comparez (Cos) et (Euc) pour chaque couleur.")
    print("- Si BERT (Euc) est très bas par rapport à BERT (Cos), c'est que la géométrie")
    print("  sphérique (Cosinus) est essentielle pour les embeddings de deep learning.")


def analyze_and_justify(stem_cos, stem_euc, lemma_cos, lemma_euc, bert_cos, bert_euc):
    
    def get_stats(df, name):
        vals = df.values[np.triu_indices_from(df, k=1)]
        return {"Méthode": name, "Moyenne": np.mean(vals), "Médiane": np.median(vals), "_vals": vals, "Max" : np.max(vals), 'Min ':np.min(vals)}

    candidates = [
        get_stats(stem_cos, "Stem (Cos)"), # get_stats(stem_euc, "Stem (Euc)"),
        get_stats(lemma_cos, "Lemma (Cos)") ,# get_stats(lemma_euc, "Lemma (Euc)"),
        #get_stats(bert_cos, "BERT (Cos)"),# get_stats(bert_euc, "BERT (Euc)")
    ]
    
    # 1. Tableau Statistique
    df_res = pd.DataFrame([{k: v for k, v in d.items() if k != '_vals'} for d in candidates])
    print("\n--- Tableau des Scores (Moyenne de similarité) ---")
    print(df_res.sort_values("Moyenne", ascending=False).to_string(index=False))
    
    # 2. Graphique Boxplot
    plt.figure(figsize=(14, 7))
    colors = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightcoral', 'lightcoral']
    bplot = plt.boxplot([d['_vals'] for d in candidates], labels=[d['Méthode'] for d in candidates], patch_artist=True)
    for patch, color in zip(bplot['boxes'], colors): patch.set_facecolor(color)
    plt.title("Comparaison : Racinisation (stem) vs Lemmatisation(lemma)", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylabel("Score de Similarité (0-1)")
    plt.show()

    # 3. JUSTIFICATION & VERDICT
    print("\n=== COMPARATIF FINAL & VERDICT ===")

    
    print("\n1. CONSTAT SUR LA DISTANCE EUCLIDIENNE :")
    print("   On observe souvent des scores artificiellement élevés avec la distance Euclidienne,")
    print("   surtout avec le Stemming (Stem Euc).")
    print("   POURQUOI ?")
    print("   - Sensibilité à la longueur : La distance euclidienne juge deux textes courts")
    print("     comme 'similaires' juste parce qu'ils sont proches de l'origine (peu de mots),")
    print("     même si leur contenu n'a rien à voir.")
    print("   - Biais du Stemming : En coupant les mots, le Stemming réduit la taille des vecteurs,")
    print("     ce qui réduit mathématiquement la distance euclidienne et gonfle le score.")
    print("   -> CONCLUSION : Nous écartons les résultats Euclidiens car ils sont biaisés par la forme.")

    print("\n2. CONSTAT SUR LA DISTANCE COSINUS :")
    print("   La mesure Cosinus évalue l'angle (le sujet) indépendamment de la longueur du texte.")
    print("   C'est la métrique standard fiable pour comparer nos documents.")

    print("\n3. VERDICT FINAL :")
    # On compare uniquement les méthodes Cosinus pour le verdict
    cos_candidates = df_res[df_res['Méthode'].str.contains("(Cos)")]
    
    # Le gagnant n'est pas forcément celui avec la plus haute moyenne, mais BERT est qualitativement meilleur.
    # Ici, on affiche le gagnant "statistique" parmi les Cosinus.
    winner = cos_candidates.loc[cos_candidates['Moyenne'].idxmax()]
    
    print(f"   Parmi les méthodes fiables (Cosinus), la méthode avec la plus forte similarité moyenne est :")
    print(f"   {winner['Méthode']} (Moyenne : {winner['Moyenne']:.4f})")
    
    print("\n   NOTE : Si BERT (Cos) a une moyenne plus faible, c'est souvent positif :")
    print("   Cela signifie qu'il est plus discriminant (il ne dit pas que tout se ressemble).")
    print("   BERT reste la méthode recommandée pour sa compréhension fine du sens.")



# --- 2. FULL PIPELINE EXECUTION ---

print("=== STEP 1: RAW TEXT RETRIEVAL ===")

# Utilisation de la variable file_input définie plus haut
docs_simple, docs_stem, docs_lemma, raw_texts = load_parquet_data(file_input, number_uni)


first_uni = list(docs_stem.keys())[0]
print(f"University: {first_uni} (Data loaded)\n")

print(f"--- Raw Text Description for: {first_uni} ---")
print(raw_texts[first_uni])



# From here, the processing already done is as follows:
# ------Lowercase: Everything is lowercase.
# ------Tokenization: Text is split into words.
# ------Punctuation Removal: Commas, dots, etc., are gone.
# ------Stop Words Removal: Words like "the", "is", "at" are ALREADY gone. 


print(" ")
print("=== STEP 2: COMPARISON OF 3 TOKENIZATION LEVELS ===")

# We directly use dictionaries filled in step 1
# No need for nltk.word_tokenize() here!

print(f"A) SIMPLE Tokens (Whole words, no stopwords):")
print(f"   {docs_simple[first_uni][:40]}...") 

print(f"\nB) LEMMATIZED Tokens (Dictionary form):")
print(f"   {docs_lemma[first_uni][:40]}...")
print("   -> Note: 'located' remains 'located' (or becomes 'locate' depending on verb/noun)")
print("   -> Note: word with 'ing' remains 'ing but when it is in a plural form 'ies', it becomes 'y'")

print(f"\nC) STEMMED Tokens (Cut roots):")
print(f"   {docs_stem[first_uni][:40]}...")
print("   -> Note: 'located' becomes 'locat' (cut short)")



# --- PARALLEL PIPELINE ---
print("---------------------------------------------------")
print("=== STEP 3: INITIAL MATRICES (Comparison of 3 methods) ===")


# 1. Creating matrices from document dictionaries
# Ensure docs_simple is not empty (see previous fixes)
td_matrix_simple = tdm_creation(docs_simple)
td_matrix_stem = tdm_creation(docs_stem)
td_matrix_lemma = tdm_creation(docs_lemma)

# 2. Displaying dimensions for comparison
print("1. SIMPLE Matrix (Tokenization only):")
print(f"   -> Dimensions: {td_matrix_simple.shape}")
# shape[1] corresponds to the number of columns, so the number of unique words
print(f"   -> Detail: {td_matrix_simple.shape[1]} unique words (MAXIMAL Vocabulary)")

print("\n2. LEMMATIZATION Matrix (Intermediate):")
print(f"   -> Dimensions: {td_matrix_lemma.shape}")
print(f"   -> Detail: {td_matrix_lemma.shape[1]} unique words")

print("\n3. STEMMING Matrix (Aggressive):")
print(f"   -> Dimensions: {td_matrix_stem.shape}")
print(f"   -> Detail: {td_matrix_stem.shape[1]} unique words (MINIMAL Vocabulary)")

print("\n(Observe how the number of columns decreases at each normalization step)")
print("-" * 40)
# ------------------------------------------------------------------


print("\n=== STEP 4.1: FILTERED MATRICES (Comparison) ===")
# Here, for this filter, we only keep elements appearing in at least N documents
filtered_stem = filter_matrix(td_matrix_stem)
filtered_lemma = filter_matrix(td_matrix_lemma)



print(f"Dimensions after filter (Stemming)      : {filtered_stem.shape}")
print(f"Dimensions after filter (Lemmatization) : {filtered_lemma.shape}")
print("-" * 40)

print("--- STEMMING RESULT ELEMENT PRESENT IN N DOCUMENTS (Extract) ---")
print(filtered_stem.iloc[:matrix_size_column, :matrix_size_line])
print("\n--- LEMMATIZATION RESULT ELEMENT PRESENT IN N DOCUMENTS (Extract) ---")
print(filtered_lemma.iloc[:matrix_size_column, :matrix_size_line])
print("\n")

print("\n=== STEP 4.2: FILTERED MATRICES (Comparison of thresold) ===")


test_threshold(td_matrix_lemma)

print("=== STEP 5: TF-IDF (Comparison) ===")
tfidf_stem = tfidf_calculation(filtered_stem)
tfidf_lemma = tfidf_calculation(filtered_lemma)

print("--- TF-IDF STEMMING (Extract) ---")
print(tfidf_stem.iloc[:matrix_size_column, :matrix_size_line])
print("\n--- TF-IDF LEMMATIZATION (Extract) ---")
print(tfidf_lemma.iloc[:matrix_size_column, :matrix_size_line])
print("\n")



print("=== STEP 6: BERT VECTORIZATION (Sentence-Transformers) ===")

# 1. Data Preparation
# IMPORTANT: BERT needs RAW text (complete sentences with context).
# We do NOT use docs_stem or docs_lemma, but 'raws_text'.
doc_names = list(raw_texts.keys())
documents = list(raw_texts.values())

# 2. Loading SBERT model
# The model will be downloaded the first time (~80MB)
# "all-MiniLM-L6-v2" is a fast and efficient model for semantic similarity.
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Computing embeddings (Vectorization)
# normalize_embeddings=True allows using dot product as cosine similarity
print("Encoding documents with BERT in progress...")
embeddings = model.encode(documents, normalize_embeddings=True)

# 4. Computing similarity matrix (Cosine)
similarity_matrix_bert = cosine_similarity(embeddings)

# 5. Creating DataFrame for display
similarity_df_bert = pd.DataFrame(
    similarity_matrix_bert,
    index=doc_names,
    columns=doc_names
)

print("--- BERT SIMILARITY (Extract) ---")
# Using your size variables to control display
# Note: It is a Documents x Documents (square) matrix
print(similarity_df_bert.iloc[:matrix_size_line, :matrix_size_line])
print("\n")


print("=== STEP 7: SIMILARITY CALCULATION (6 METHODS) ===")

# --- CALCUL DES 6 MÉTHODES DE SIMILARITÉ ---
print("\n=== CALCUL DES SIMILARITÉS (6 COMBINAISONS) ===")

# 1. Stemming
sim_stem_cos = pd.DataFrame(cosine_similarity(tfidf_stem), index=doc_names, columns=doc_names)
sim_stem_euc = pd.DataFrame(dist_to_sim(euclidean_distances(tfidf_stem)), index=doc_names, columns=doc_names)

# 2. Lemmatization
sim_lemma_cos = pd.DataFrame(cosine_similarity(tfidf_lemma), index=doc_names, columns=doc_names)
sim_lemma_euc = pd.DataFrame(dist_to_sim(euclidean_distances(tfidf_lemma)), index=doc_names, columns=doc_names)

# 3. BERT
sim_bert_cos = pd.DataFrame(cosine_similarity(embeddings), index=doc_names, columns=doc_names)
sim_bert_euc = pd.DataFrame(dist_to_sim(euclidean_distances(embeddings)), index=doc_names, columns=doc_names)




analyze_and_justify(sim_stem_cos, sim_stem_euc, sim_lemma_cos, sim_lemma_euc, sim_bert_cos, sim_bert_euc)



# mapping country into region for the descriptive analysis

country_to_region = {
    # Europe
    "United Kingdom": "Europe",
    "Switzerland": "Europe",
    "Germany": "Europe",
    "Sweden": "Europe",
    "Belgium": "Europe",
    "France": "Europe",
    "Netherlands": "Europe",
    "Denmark": "Europe",
    "Finland": "Europe",
    "Norway": "Europe",
    "Spain": "Europe",
    "Ireland": "Europe",
    "Austria": "Europe",
    "Italy": "Europe",
    "Russian Federation": "Europe",
    "Czechia": "Europe",
    "Greece": "Europe",
    "Portugal": "Europe",
    "Cyprus": "Europe",
    "Poland": "Europe",
    "Iceland": "Europe",
    "Hungary": "Europe",
    "Luxembourg": "Europe",
    "Estonia": "Europe",
    "Romania": "Europe",
    "Slovenia": "Europe",
    "Malta": "Europe",
    "Lithuania": "Europe",
    "Latvia": "Europe",
    "Ukraine": "Europe",
    "Serbia": "Europe",
    "Slovakia": "Europe",
    "Croatia": "Europe",
    "Bulgaria": "Europe",
    "North Macedonia": "Europe",
    "Bosnia and Herzegovina": "Europe",
    "Kosovo": "Europe",

    # North America
    "United States": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    "Costa Rica": "North America",
    "Jamaica": "North America",
    "Cuba": "North America",

    # Asia (incluant Turquie, Arménie, Kazakhstan, Fiji comme demandé)
    "China": "Asia",
    "Singapore": "Asia",
    "Japan": "Asia",
    "Hong Kong": "Asia",
    "South Korea": "Asia",
    "Taiwan": "Asia",
    "Macao": "Asia",
    "Saudi Arabia": "Asia",
    "United Arab Emirates": "Asia",
    "India": "Asia",
    "Qatar": "Asia",
    "Malaysia": "Asia",
    "Lebanon": "Asia",
    "Turkey": "Asia",
    "Iran": "Asia",
    "Brunei Darussalam": "Asia",
    "Jordan": "Asia",
    "Bahrain": "Asia",
    "Kazakhstan": "Asia",
    "Pakistan": "Asia",
    "Thailand": "Asia",
    "Uzbekistan": "Asia",
    "Vietnam": "Asia",
    "Kuwait": "Asia",
    "Bangladesh": "Asia",
    "Indonesia": "Asia",
    "Philippines": "Asia",
    "Sri Lanka": "Asia",
    "Fiji": "Asia",
    "Georgia": "Asia",
    "Armenia": "Asia",
}


# --- 8. FILTRAGE ET SAUVEGARDE FINALE ---



print("\n=== STEP 8: SAUVEGARDE (Fichier JSON Uniqu - Top TF-IDF) ===")

#number of the most significant tokens to keep per university (based on TF-IDF Score)
TOP_K = 25

# 1. Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory created: {output_dir}")

#2. Token Selection (Top TF-IDF)
docs_lemma_clean = {}

# 'tfidf_lemma' is your DataFrame (Index=University Name, Columns=Tokens, Values=TF-IDF Score)
print("Selecting top-k signature tokens for each university...")
for univ_name, row in tfidf_lemma.iterrows():
    # Sort and keep the TOP_K tokens with the highest TF-IDF scores
    top_tokens_series = row.nlargest(TOP_K)
    
    # Filter out tokens with score <= 0 (just in case K > actual vocab size)
    valid_top_tokens = top_tokens_series[top_tokens_series > 0].index.tolist()
    
    # Store in dictionary
    docs_lemma_clean[univ_name] = valid_top_tokens

# --- Metadata Integration (Regions) ---
# 3. Reload original Parquet to retrieve metadata (country, region, etc.)
df_meta = pd.read_parquet(file_input)

# Make sure name is string and aligned
df_meta["name"] = df_meta["name"].astype(str).str.strip()

# 4. Build region mapping depending on mode
if CURRENT_MODE.lower() in ["the", "the_2021"]:
    # Ensure country is clean
    df_meta["country"] = df_meta["country"].astype(str).str.strip()
    # country → region, others = "Other"
    df_meta["region"] = df_meta["country"].map(country_to_region).fillna("Other")
else:
    # If region already exists (e.g. QS), keep it, otherwise set to "Other"
    if "region" in df_meta.columns:
        df_meta["region"] = df_meta["region"].astype(str).str.strip().replace("", "Other")
    else:
        df_meta["region"] = "Other"

# Dict {university_name: region} aligned with docs_lemma_clean keys
regions = (
    df_meta.set_index("name")["region"]
           .reindex(docs_lemma_clean.keys())
           .fillna("Other")
           .to_dict()
)


# --- Final JSON Assembly ---
# 5. Build export structure
data_export = {
    "info": {
        "mode": CURRENT_MODE,
        "nb_universites": len(docs_lemma_clean),
        "nb_mots_vocab_total": tfidf_lemma.shape[1], # Total vocabulary size
        "top_k_tokens_per_univ": TOP_K
    },
    "tokens": docs_lemma_clean,
    "regions": regions,
    # "matrice": filtered_lemma.to_dict(orient="index") # Commented out: No need to export the full dense matrix
}

# 6. Custom JSON Encoder for NumPy data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# 7. Save to JSON file
print(f"Saving to: {file_output} ...")
with open(file_output, 'w', encoding='utf-8') as f:
    json.dump(data_export, f, cls=NumpyEncoder, ensure_ascii=False, indent=4)

print("Done! JSON file successfully generated with Top-TFIDF tokens.")
