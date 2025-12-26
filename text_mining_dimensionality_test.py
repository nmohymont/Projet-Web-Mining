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
import pickle # Librairie standard pour sauvegarder des variables


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
number_uni = 150
# Data file path
file_the = 'DATA/PARQUET/the_university_corpus.parquet'
file_qs = 'DATA/PARQUET/qs_university_corpus.parquet'

# --- STEMMING CONFIGURATION ---
stemmer = nltk.stem.SnowballStemmer("english")

# --- LEMMATIZATION CONFIGURATION ---
lemmatizer = nltk.stem.WordNetLemmatizer()

# --- STOP WORDS ---
stop_words = list(set(stopwords.words('english'))) + ["'s", "also", "one", "two",]
# stopwords.words('english') => list provided by NLTK library containing common English words with no significant meaning ("the", "is", "at", "which", "on", "a", "an", "and"...)
# set => Removes duplicates
# ["'s", "also", "thi", "one", "two",] => Adding context-specific words => Our own rules

def extract_tokens(text, mode='stem'):
    # ... (start of function unchanged) ...
    text = str(text).lower() # Lowercase 

    # Radical option: remove English contractions => '
    # We completely remove 's, 're, 've, n't BEFORE tokenization
    text = text.replace("’", "'")  # standardize curly quotes to straight aspostrophes
    # This transforms "university's" into "university" because a token can be university's so the stopwords cannot be detected
    text = re.sub(r"'\w+", '', text)

    tokens = nltk.word_tokenize(text) 
    processed_tokens = []
    for t in tokens:
        if t.isalpha() and t not in string.punctuation and t not in stop_words : # WITHOUT numbers, punctuation, and stop words 
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

def load_parquet_data(filename, number_uni=5): # Par défaut ! Pas correct car on ne connait pas à l'avance le nombre total d'universités dans le fichier
    df = pd.read_parquet(filename) 
    if number_uni:
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


# --- 2. FULL PIPELINE EXECUTION ---

print("=== STEP 1: RAW TEXT RETRIEVAL ===")


#""""""""""""""""""""""""
# ATTENTION, this is where we choose the number of documents to process!!!!
#"""""""""""""""""""""""
docs_simple, docs_stem, docs_lemma, raw_texts = load_parquet_data(file_qs, number_uni)
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

print("\n--- GENERATING HEATMAP ---")

'''
plt.figure(figsize=(8, 6))
plt.imshow(similarity_df_bert, interpolation='nearest', cmap='viridis')
plt.title('BERT Semantic Similarity Matrix', fontsize=14, fontweight='bold')
plt.colorbar(label='Similarity Score (0 to 1)')

# Ajout des labels
plt.xticks(range(len(doc_names)), doc_names, rotation=45, ha='right')
plt.yticks(range(len(doc_names)), doc_names)

# Ajout des valeurs dans les cases
for i in range(len(doc_names)):
    for j in range(len(doc_names)):
        val = similarity_df_bert.iloc[i, j]
        color = 'black' if val > 0.6 else 'white' # Contraste pour lisibilité
        plt.text(j, i, f"{val:.2f}", ha='center', va='center', color=color)

plt.tight_layout()
#Add this line to show the plot
#plt.show()

'''




print("=== STEP 7: SIMILARITY MATRIX (Comparison) ===")
sim_stem = pd.DataFrame(cosine_similarity(tfidf_stem), index=tfidf_stem.index, columns=tfidf_stem.index)
sim_lemma = pd.DataFrame(cosine_similarity(tfidf_lemma), index=tfidf_lemma.index, columns=tfidf_lemma.index)

print("--- STEMMING SIMILARITY (5x5) ---")
print(sim_stem.iloc[:matrix_size_column, :matrix_size_line])
print("\n--- LEMMATIZATION SIMILARITY (5x5) ---")
print(sim_lemma.iloc[:matrix_size_column, :matrix_size_line])
print("\n")

print("=== STEP 8: COMPARATIVE DISPLAY OF 2 PLOTS ===")


#Affiche des matrices de similarité côte à côte pour comparaison visuelle

#plot_comparison(sim_stem, sim_lemma)


print("\n=== STEP 9: STATISTICAL COMPARISON & FINAL VERDICT ===")

def analyze_and_compare_methods(df_stem, df_lemma, df_bert):
    """
    Fonction complète qui calcule les stats, affiche le graphe 
    et désigne la meilleure méthode.
    """
    
    # --- FONCTION INTERNE DE CALCUL ---
    def get_stats(df, name):
        matrix = df.values
        # Indices du triangle supérieur (sans la diagonale)
        upper_indices = np.triu_indices_from(matrix, k=1)
        values = matrix[upper_indices]
        
        return {
            "Méthode": name,
            "Moyenne Sim.": np.mean(values),
            "Médiane Sim.": np.median(values),
            "Écart-Type": np.std(values),
            "Max Sim.": np.max(values),
            "_values": values # On garde les valeurs brutes pour le graphique
        }

    # --- CALCUL DES DONNÉES ---
    stats_list = [
        get_stats(df_stem, "Stemming (Racine)"),
        get_stats(df_lemma, "Lemmatization (Dico)"),
        get_stats(df_bert, "BERT (Sémantique)")
    ]
    
    # Création du DataFrame pour l'affichage (sans la colonne _values qui est trop lourde)
    df_display = pd.DataFrame([{k: v for k, v in d.items() if k != '_values'} for d in stats_list])
    
    print("--- 1. Comparaison des Scores ---")
    print(df_display.round(4).to_string(index=False))

    # --- GÉNÉRATION DU GRAPHIQUE (BOXPLOT) ---
    plt.figure(figsize=(12, 6))
    
    # Récupération des valeurs brutes stockées
    data_to_plot = [d['_values'] for d in stats_list]
    labels = [d['Méthode'] for d in stats_list]
    
    plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor="lightblue"))
    
    plt.title("Distribution de la Similarité entre Universités (Dispersion)", fontsize=14)
    plt.ylabel("Score de Similarité Cosinus (0 à 1)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # --- JUGEMENT ---
    print("\n--- 2. VERDICT FINAL : QUELLE EST LA MEILLEURE MÉTHODE ? ---")
    
    # Critère : La méthode avec la moyenne la plus élevée (capte le mieux le fond)
    # On exclut si la moyenne est > 0.99 (ce qui serait suspect/bug)
    best_row_idx = df_display['Moyenne Sim.'].idxmax()
    winner = df_display.iloc[best_row_idx]
    
    name = winner['Méthode']
    score = winner['Moyenne Sim.']
    
    print(f"La meilleur méthode est : {name.upper()}")
    print(f"   -> Score Moyen : {score:.4f}")
    
   

analyze_and_compare_methods(sim_stem, sim_lemma, similarity_df_bert)


#save processed data for external analysis

print("\n=== SAUVEGARDE DES DONNÉES POUR ANALYSE EXTERNE ===")
# On sauvegarde les deux variables essentielles pour vos graphiques :
# 1. docs_lemma (dictionnaire des tokens pour l'analyse thématique)
# 2. td_matrix_lemma (matrice pour le nuage de mots)

with open('DATA-CLEANED/donnees_traitees.pkl', 'wb') as f:
    pickle.dump((docs_lemma, td_matrix_lemma), f)

print("Succès ! Les données sont sauvegardées dans 'donnees_traitees.pkl'.")
print("Vous pouvez maintenant lancer le fichier de visualisation.")
