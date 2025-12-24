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
from wordcloud import WordCloud # Nuage de mot 
from nltk.sentiment import SentimentIntensityAnalyzer #Analyse de sentiment 


# --- 1. CONFIGURATION AND TOOLS ---

# Note: Uncomment the lines below during the first execution
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') 
nltk.download('omw-1.4')


# Variables 
#------------------
# Minimum words to keep a column in the term-document matrix
Numbre_Min_words = 3
# Matrix extract size to display
Matrix_size_column = 10
Matrix_size_line = 10
# Number of universities to process (Set to None to process all)
number_uni = 150

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
        if t.isalpha() and t not in string.punctuation and t not in stop_words and : # WITHOUT numbers, punctuation, and stop words 
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

def load_parquet_data(filename, number_uni=5):
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


# --- UTILITY FUNCTIONS ---
# Term-document Matrix
def tdm_creation(documents_dict):
    vocabulaire = set(token for tokens in documents_dict.values() for token in tokens)
    term_frequencies = {doc: Counter(tokens) for doc, tokens in documents_dict.items()}
    matrix = pd.DataFrame(
        {term: [term_frequencies[doc].get(term, 0) for doc in documents_dict] for term in vocabulaire},
        index=documents_dict.keys()
    )
    return matrix

# Add another filter
def filter_matrix(matrix):
    doc_freq_filter = (matrix > 0).sum(axis=0)   
    return matrix.loc[:, doc_freq_filter >= Numbre_Min_words]

def tfidf_calculation(filtered_matrix):
    row_sums = filtered_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1 
    tf = filtered_matrix.div(row_sums, axis=0)
    df_count = (filtered_matrix > 0).sum(axis=0)
    N = filtered_matrix.shape[0]
    idf = np.log(N / df_count)
    return tf.mul(idf, axis=1)


# --- 2. FULL PIPELINE EXECUTION ---

print("=== STEP 1: RAW TEXT RETRIEVAL ===")


#""""""""""""""""""""""""
# ATTENTION, this is where we choose the number of documents to process!!!!
#""""""""""""""""""""""""

docs_simple, docs_stem, docs_lemma, textes_bruts = load_parquet_data('DATA/PARQUET/the_university_corpus.parquet', number_uni)
premier_titre = list(docs_stem.keys())[0]
print(f"University: {premier_titre} (Data loaded)\n")

print(f"--- Raw Text Description for: {premier_titre} ---")
print(textes_bruts[premier_titre])



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
print(f"   {docs_simple[premier_titre][:20]}...") 

print(f"\nB) LEMMATIZED Tokens (Dictionary form):")
print(f"   {docs_lemma[premier_titre][:20]}...")
print("   -> Note: 'located' remains 'located' (or becomes 'locate' depending on verb/noun)")

print(f"\nC) STEMMED Tokens (Cut roots):")
print(f"   {docs_stem[premier_titre][:20]}...")
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


print("\n=== STEP 4: FILTERED MATRICES (Comparison) ===")
# Here, for this filter, we only keep elements appearing in at least N documents
filtered_stem = filter_matrix(td_matrix_stem)
filtered_lemma = filter_matrix(td_matrix_lemma)

print(f"Dimensions after filter (Stemming)      : {filtered_stem.shape}")
print(f"Dimensions after filter (Lemmatization) : {filtered_lemma.shape}")
print("-" * 40)

print("--- STEMMING RESULT ELEMENT PRESENT IN N DOCUMENTS (Extract) ---")
print(filtered_stem.iloc[:Matrix_size_column, :Matrix_size_line])
print("\n--- LEMMATIZATION RESULT ELEMENT PRESENT IN N DOCUMENTS (Extract) ---")
print(filtered_lemma.iloc[:Matrix_size_column, :Matrix_size_line])
print("\n")

print("=== STEP 5: TF-IDF (Comparison) ===")
tfidf_stem = tfidf_calculation(filtered_stem)
tfidf_lemma = tfidf_calculation(filtered_lemma)

print("--- TF-IDF STEMMING (Extract) ---")
print(tfidf_stem.iloc[:Matrix_size_column, :Matrix_size_line])
print("\n--- TF-IDF LEMMATIZATION (Extract) ---")
print(tfidf_lemma.iloc[:Matrix_size_column, :Matrix_size_line])
print("\n")



print("=== STEP 6: BERT VECTORIZATION (Sentence-Transformers) ===")

# 1. Data Preparation
# IMPORTANT: BERT needs RAW text (complete sentences with context).
# We do NOT use docs_stem or docs_lemma, but 'textes_bruts'.
doc_names = list(textes_bruts.keys())
documents = list(textes_bruts.values())

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
print(similarity_df_bert.iloc[:Matrix_size_line, :Matrix_size_line])
print("\n")

print("=== STEP 7: SIMILARITY MATRIX (Comparison) ===")
sim_stem = pd.DataFrame(cosine_similarity(tfidf_stem), index=tfidf_stem.index, columns=tfidf_stem.index)
sim_lemma = pd.DataFrame(cosine_similarity(tfidf_lemma), index=tfidf_lemma.index, columns=tfidf_lemma.index)

print("--- STEMMING SIMILARITY (5x5) ---")
print(sim_stem.iloc[:Matrix_size_column, :Matrix_size_line])
print("\n--- LEMMATIZATION SIMILARITY (5x5) ---")
print(sim_lemma.iloc[:Matrix_size_column, :Matrix_size_line])
print("\n")

print("=== STEP 8: COMPARATIVE DISPLAY OF 2 PLOTS ===")

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

#plot_comparison(sim_stem, sim_lemma)

print("\n=== STEP 9: WORD CLOUD (Top 15 Words) ===")

# 1. Calculate total frequency for each word across all documents
# We use the Lemmatized matrix for better readability
word_frequencies = td_matrix_lemma.sum(axis=0).to_dict()

# 2. Generate the Word Cloud
# max_words=15 limits the display to the top 15 most frequent words
wc = WordCloud(width=800, height=400, max_words=50, background_color='white')
wc.generate_from_frequencies(word_frequencies)

# 3. Display the plot
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off') # Hide axes
plt.title('Top 50 Most Frequent Words (Lemmatized)', fontsize=16)
plt.show()

print("\n=== STEP 11: THEMATIC ANALYSIS (Excellence, Innovation, Inclusion) ===")

# 1. Define Keywords for each specific Theme (The "Dictionary")
# We use root words/lemmas because we are searching in 'docs_lemma'
themes_lexicon = {
    "Excellence": [
        "excellence", "leading", "prestigious", "top", "rank", "award", 
        "elite", "quality", "leader", "best", "world-class", "reputation",
        "outstanding", "achievement", "merit"
    ],
    "Innovation": [
        "innovation", "research", "technology", "science", "discovery", 
        "create", "modern", "future", "digital", "cutting-edge", 
        "develop", "lab", "creative", "new", "advance"
    ],
    "Inclusion": [
        "community", "diverse", "inclusion", "support", "social", 
        "equality", "access", "aid", "scholarship", "welcome", 
        "opportunity", "global", "public", "help", "together"
    ]
}

# 2. Initialize Counters
# theme_counts: Total score for each theme across all docs
# specific_word_counts: Which specific words are used most?
theme_counts = {theme: 0 for theme in themes_lexicon}
specific_word_counts = {theme: {} for theme in themes_lexicon}

# 3. Analyze the Corpus (Using Lemma tokens for better matching)
# We use docs_lemma because "located" became "located" but "universities" became "university"
for title, tokens in docs_lemma.items():
    for token in tokens:
        for theme, keywords in themes_lexicon.items():
            if token in keywords:
                # Increment Global Theme Score
                theme_counts[theme] += 1
                
                # Increment Specific Word Count within that theme
                if token in specific_word_counts[theme]:
                    specific_word_counts[theme][token] += 1
                else:
                    specific_word_counts[theme][token] = 1

# --- A. DISPLAY RESULTS ---
print("\n--- Thematic Scores (Total mentions in Corpus) ---")
for theme, score in theme_counts.items():
    print(f"{theme}: {score}")

print("\n--- Detailed Word Frequency per Theme (Top 3) ---")
for theme, words_dict in specific_word_counts.items():
    # Sort by frequency
    sorted_words = sorted(words_dict.items(), key=lambda item: item[1], reverse=True)
    top_words = sorted_words[:3] # Get top 3
    print(f"> {theme}: {top_words}")

# --- B. VISUALIZATION ---
# Create a bar chart comparing the 3 themes
plt.figure(figsize=(10, 6))

themes = list(theme_counts.keys())
scores = list(theme_counts.values())
colors = ['gold', 'cyan', 'orchid'] # Gold for Excellence, Cyan for Innovation, Orchid for Inclusion

plt.bar(themes, scores, color=colors, edgecolor='black')

plt.title('Analysis of University Values (Tonality)', fontsize=16)
plt.xlabel('Themes', fontsize=12)
plt.ylabel('Frequency of Terms', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for i, v in enumerate(scores):
    plt.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

plt.show()