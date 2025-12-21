import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURATION ET OUTILS ---

# Note : Décommentez les lignes ci-dessous lors de la première exécution
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet') 
# nltk.download('omw-1.4')

# --- CONFIGURATION RACINISATON (Stemming) ---
stemmer = nltk.stem.SnowballStemmer("english")

# --- CONFIGURATION LEMMATISATION ---
lemmatizer = nltk.stem.WordNetLemmatizer()

# --- STOP WORDS ---
stop_words = list(set(stopwords.words('english'))) + ["'s", "also", "thi", "one", "two"]

def extract_tokens(text, mode='stem'):
    """
    Fonction de nettoyage NLP complète.
    mode: 'stem' pour racinisation, 'lemma' pour lemmatisation
    """
    if pd.isna(text) or text == "":
        return []
    
    # 1. Normalisation
    text = str(text).lower()
    
    # 2. Tokenisation
    tokens = nltk.word_tokenize(text)
    
    processed_tokens = []
    for t in tokens:
        # 3. Filtrage
        if t not in string.punctuation and t not in stop_words:
            
            # 4. CHOIX DU TRAITEMENT
            if mode == 'stem':
                processed_tokens.append(stemmer.stem(t))
            elif mode == 'lemma':
                processed_tokens.append(lemmatizer.lemmatize(t))
            
    return processed_tokens

def recuperer_documents(fichier, nombre_docs=5):
    """
    Charge le dataset et prépare DEUX versions des données.
    """
    df = pd.read_parquet(fichier) 
    
    if nombre_docs:
        df = df.iloc[:nombre_docs]

    docs_stem = {}      
    docs_lemma = {}     
    raw_texts = {} 
    
    for _, row in df.iterrows():
        titre = row['name']
        desc = row['description']
        
        docs_stem[titre] = extract_tokens(desc, mode='stem')
        docs_lemma[titre] = extract_tokens(desc, mode='lemma')
        raw_texts[titre] = desc 
        
    return docs_stem, docs_lemma, raw_texts

# --- 2. EXÉCUTION DU PIPELINE COMPLET ---

print("=== ÉTAPE 1 : RÉCUPERATION DU TEXTE BRUT ===")


#""""""""""""""""""""""""
#ATTENTION, c'est ici qu'on choisi le nombre de documents à traiter !!!!
#""""""""""""""""""""""""
docs_stem, docs_lemma, textes_bruts = recuperer_documents('the_university_corpus.parquet', nombre_docs=10)

premier_titre = list(docs_stem.keys())[0]
print(f"Université : {premier_titre}")
print(f"Texte brut (extrait) : {str(textes_bruts[premier_titre])[:100]}...\n")

print("=== ÉTAPE 2 : COMPARAISON TOKENISATION ===")
tokens_bruts = nltk.word_tokenize(str(textes_bruts[premier_titre]).lower())

print(f"A) Tokens BRUTS (5 premiers)      : {tokens_bruts[:5]}")
print(f"B) Tokens RACINISÉS (Stemming)    : {docs_stem[premier_titre][:5]}")
print(f"C) Tokens LEMMATISÉS (Lemmatize)  : {docs_lemma[premier_titre][:5]}")


# --- FONCTIONS UTILITAIRES ---
def creer_matrice(documents_dict):
    vocabulaire = set(token for tokens in documents_dict.values() for token in tokens)
    term_frequencies = {doc: Counter(tokens) for doc, tokens in documents_dict.items()}
    matrix = pd.DataFrame(
        {term: [term_frequencies[doc].get(term, 0) for doc in documents_dict] for term in vocabulaire},
        index=documents_dict.keys()
    )
    return matrix

def filtrer_matrice(matrix):
    doc_freq_filter = (matrix > 0).sum(axis=0)   
    return matrix.loc[:, doc_freq_filter >= 2]

def calcul_tfidf(filtered_matrix):
    row_sums = filtered_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1 
    tf = filtered_matrix.div(row_sums, axis=0)
    df_count = (filtered_matrix > 0).sum(axis=0)
    N = filtered_matrix.shape[0]
    idf = np.log(N / df_count)
    return tf.mul(idf, axis=1)

# --- PIPELINE PARALLÈLE ---

print("\n=== ÉTAPE 3 : MATRICES INITIALES (Comparaison des dimensions) ===")
td_matrix_stem = creer_matrice(docs_stem)
td_matrix_lemma = creer_matrice(docs_lemma)

# --- MODIFICATION DEMANDÉE : AFFICHAGE DÉTAILLÉ DES DIMENSIONS ---
print("1. Matrice issue de la RACINISATION (Stemming) :")
print(f"   -> Dimensions totales : {td_matrix_stem.shape}")
print(f"   -> Détail : {td_matrix_stem.shape[0]} documents x {td_matrix_stem.shape[1]} mots uniques (vocabulaire)")

print("\n2. Matrice issue de la LEMMATISATION :")
print(f"   -> Dimensions totales : {td_matrix_lemma.shape}")
print(f"   -> Détail : {td_matrix_lemma.shape[0]} documents x {td_matrix_lemma.shape[1]} mots uniques (vocabulaire)")

print("\n(Note : Le stemming a souvent moins de colonnes car il regroupe plus de mots ensemble)")
# ------------------------------------------------------------------


print("\n=== ÉTAPE 4 : MATRICES FILTRÉES (Comparaison) ===")
filtered_stem = filtrer_matrice(td_matrix_stem)
filtered_lemma = filtrer_matrice(td_matrix_lemma)

print(f"Dimensions après filtre (Stemming)      : {filtered_stem.shape}")
print(f"Dimensions après filtre (Lemmatisation) : {filtered_lemma.shape}")
print("-" * 40)

print("--- RÉSULTAT STEMMING (Extrait) ---")
print(filtered_stem.iloc[:5, :5])
print("\n--- RÉSULTAT LEMMATISATION (Extrait) ---")
print(filtered_lemma.iloc[:5, :5])
print("\n")

print("=== ÉTAPE 5 : TF-IDF (Comparaison) ===")
tfidf_stem = calcul_tfidf(filtered_stem)
tfidf_lemma = calcul_tfidf(filtered_lemma)

print("--- TF-IDF STEMMING (Extrait) ---")
print(tfidf_stem.iloc[:5, :5])
print("\n--- TF-IDF LEMMATISATION (Extrait) ---")
print(tfidf_lemma.iloc[:5, :5])
print("\n")

print("=== ÉTAPE 6 : MATRICE DE SIMILARITÉ (Comparaison) ===")
sim_stem = pd.DataFrame(cosine_similarity(tfidf_stem), index=tfidf_stem.index, columns=tfidf_stem.index)
sim_lemma = pd.DataFrame(cosine_similarity(tfidf_lemma), index=tfidf_lemma.index, columns=tfidf_lemma.index)

print("--- SIMILARITÉ STEMMING (5x5) ---")
print(sim_stem.iloc[:5, :5])
print("\n--- SIMILARITÉ LEMMATISATION (5x5) ---")
print(sim_lemma.iloc[:5, :5])
print("\n")

print("=== ÉTAPE 7 : AFFICHAGE COMPARATIF DES 2 PLOTS ===")

def plot_comparison(df_stem, df_lemma):
    # Création d'une figure avec 1 ligne et 2 colonnes
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Configuration des données à itérer
    data_list = [
        (axes[0], df_stem, "Similarité - STEMMING (Racines coupées)"),
        (axes[1], df_lemma, "Similarité - LEMMATISATION (Mots réels)")
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

plot_comparison(sim_stem, sim_lemma)