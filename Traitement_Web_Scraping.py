import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer # Pour BERT

# --- 1. CONFIGURATION ET OUTILS ---

# Note : Décommentez les lignes ci-dessous lors de la première exécution
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet') 
# nltk.download('omw-1.4')


#Variable 
#------------------
#  Minimum de mots pour garder une colonne dans la matrice terme-document
Numbre_Min_words = 3
# Taille des extraits de matrices à afficher
Matrix_size_column = 10
Matrix_size_line = 10
#Nombre documents à traiter (Mettre None pour tout traiter)
nombre_docs = 10

# --- CONFIGURATION RACINISATON (Stemming) ---
stemmer = nltk.stem.SnowballStemmer("english")

# --- CONFIGURATION LEMMATISATION ---
lemmatizer = nltk.stem.WordNetLemmatizer()

# --- STOP WORDS ---
stop_words = list(set(stopwords.words('english'))) + ["'s", "also", "one", "two",]
#stopwords.words('english' => liste fournie par la librairie NLTK qui contient les mots les plus courants de la langue anglaise qui n'apportent pas de sens ( "the", "is", "at", "which", "on", "a", "an", "and"...)
#set => Supprime les doublons
#["'s", "also", "thi", "one", "two",] => Ajout de mots spécifiques à notre contexte => Nos propres règles
def extract_tokens(text, mode='stem'):
    # ... (début de la fonction inchangé) ...
    text = str(text).lower() #Minuscule 

    #Option radicale : suppression des contractions anglaises => '
    #On supprime carrément les 's, 're, 've, n't AVANT tokenisation
    text = text.replace("’", "'") 
    # Cela transforme "university's" en "university"
    text = re.sub(r"'\w+", '', text)

    tokens = nltk.word_tokenize(text) 
    processed_tokens = []
    for t in tokens:
        if t not in string.punctuation and t not in stop_words and t.isalpha(): # SANS stop words et ponctuation et chiffre 
            if mode == 'stem':
                processed_tokens.append(stemmer.stem(t))
            elif mode == 'lemma':
                processed_tokens.append(lemmatizer.lemmatize(t))
            elif mode == 'none':  
                processed_tokens.append(t) # On ajoute le mot sans le modifier
    return processed_tokens

# Ce qui a été traité dans "extract_tokens" :

# ------Mise en minuscule (Lowercase) : Tout est en minuscule.
# ------Tokenisation : Le texte est découpé en mots.
#-------Suppression de la Ponctuation : Les virgules, points, etc. sont partis.
# ------Suppression des Stop Words : Les mots comme "the", "is", "at" sont DÉJÀ partis. 

def recuperer_documents(fichier, nombre_docs=5):
    df = pd.read_parquet(fichier) 
    if nombre_docs:
        df = df.iloc[:nombre_docs]
    
    docs_simple = {}    #Tokenisation seule
    docs_stem = {}      
    docs_lemma = {}     
    raw_texts = {} 
    
    for _, row in df.iterrows():
        titre = row['name']
        desc = row['description']

              
        docs_simple[titre] = extract_tokens(desc, mode='none') 
        docs_stem[titre] = extract_tokens(desc, mode='stem')
        docs_lemma[titre] = extract_tokens(desc, mode='lemma')
        raw_texts[titre] = desc 
        
    return docs_simple, docs_stem, docs_lemma, raw_texts


# --- FONCTIONS UTILITAIRES ---
def creer_matrice(documents_dict):
    vocabulaire = set(token for tokens in documents_dict.values() for token in tokens)
    term_frequencies = {doc: Counter(tokens) for doc, tokens in documents_dict.items()}
    matrix = pd.DataFrame(
        {term: [term_frequencies[doc].get(term, 0) for doc in documents_dict] for term in vocabulaire},
        index=documents_dict.keys()
    )
    return matrix

#Rajouter autre filtre
def filtrer_matrice(matrix):
    doc_freq_filter = (matrix > 0).sum(axis=0)   
    return matrix.loc[:, doc_freq_filter >= Numbre_Min_words]

def calcul_tfidf(filtered_matrix):
    row_sums = filtered_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1 
    tf = filtered_matrix.div(row_sums, axis=0)
    df_count = (filtered_matrix > 0).sum(axis=0)
    N = filtered_matrix.shape[0]
    idf = np.log(N / df_count)
    return tf.mul(idf, axis=1)


# --- 2. EXÉCUTION DU PIPELINE COMPLET ---

print("=== ÉTAPE 1 : RÉCUPERATION DU TEXTE BRUT ===")


#""""""""""""""""""""""""
#ATTENTION, c'est ici qu'on choisi le nombre de documents à traiter !!!!
#""""""""""""""""""""""""

docs_simple, docs_stem, docs_lemma, textes_bruts = recuperer_documents('the_university_corpus.parquet', nombre_docs)
premier_titre = list(docs_stem.keys())[0]
print(f"Université : {premier_titre} (Données chargées)\n")

print(f"--- Description brute (Raw Text) pour : {premier_titre} ---")
print(textes_bruts[premier_titre])



#A partir d'ici, le traitement qui a déjà été fait est le suivant :
# ------Mise en minuscule (Lowercase) : Tout est en minuscule.
# ------Tokenisation : Le texte est découpé en mots.
#-------Suppression de la Ponctuation : Les virgules, points, etc. sont partis.
# ------Suppression des Stop Words : Les mots comme "the", "is", "at" sont DÉJÀ partis. 


print(" ")
print("=== ÉTAPE 2 : COMPARAISON DES 3 NIVEAUX DE TOKENISATION ===")

# On utilise directement les dictionnaires déjà remplis à l'étape 1
# Plus besoin de nltk.word_tokenize() ici !

print(f"A) Tokens SIMPLES (Mots entiers, sans stopwords) :")
print(f"   {docs_simple[premier_titre][:20]}...") 

print(f"\nB) Tokens LEMMATISÉS (Forme dictionnaire) :")
print(f"   {docs_lemma[premier_titre][:20]}...")
print("   -> Notez : 'located' reste 'located' (ou devient 'locate' selon le verbe/nom)")

print(f"\nC) Tokens RACINISÉS (Stemming - Racines coupées) :")
print(f"   {docs_stem[premier_titre][:20]}...")
print("   -> Notez : 'located' devient 'locat' (coupé net)")



# --- PIPELINE PARALLÈLE ---
print("---------------------------------------------------")
print("=== ÉTAPE 3 : MATRICES INITIALES (Comparaison des 3 méthodes) ===")


# 1. Création des matrices à partir des dictionnaires de documents
# Assure-toi que docs_simple n'est pas vide (voir correctifs précédents)
td_matrix_simple = creer_matrice(docs_simple)
td_matrix_stem = creer_matrice(docs_stem)
td_matrix_lemma = creer_matrice(docs_lemma)

# 2. Affichage des dimensions pour comparaison
print("1. Matrice SIMPLE (Juste Tokenisation) :")
print(f"   -> Dimensions : {td_matrix_simple.shape}")
# shape[1] correspond au nombre de colonnes, donc au nombre de mots uniques
print(f"   -> Détail : {td_matrix_simple.shape[1]} mots uniques (Vocabulaire MAXIMAL)")

print("\n2. Matrice LEMMATISATION (Intermédiaire) :")
print(f"   -> Dimensions : {td_matrix_lemma.shape}")
print(f"   -> Détail : {td_matrix_lemma.shape[1]} mots uniques")

print("\n3. Matrice RACINISATION (Stemming - Agressif) :")
print(f"   -> Dimensions : {td_matrix_stem.shape}")
print(f"   -> Détail : {td_matrix_stem.shape[1]} mots uniques (Vocabulaire MINIMAL)")

print("\n(Observez comment le nombre de colonnes diminue à chaque étape de normalisation)")
print("-" * 40)
# ------------------------------------------------------------------


print("\n=== ÉTAPE 4 : MATRICES FILTRÉES (Comparaison) ===")
#Ici, pour ce filtre, on ne traite garde que les éléments apparaissant dans au moins N documents
filtered_stem = filtrer_matrice(td_matrix_stem)
filtered_lemma = filtrer_matrice(td_matrix_lemma)

print(f"Dimensions après filtre (Stemming)      : {filtered_stem.shape}")
print(f"Dimensions après filtre (Lemmatisation) : {filtered_lemma.shape}")
print("-" * 40)

print("--- RÉSULTAT STEMMING ELEMENT PRESENT DANS N DOCUMENTS(Extrait) ---")
print(filtered_stem.iloc[:Matrix_size_column, :Matrix_size_line])
print("\n--- RÉSULTAT LEMMATISATION ELEMENT PRESENT DANS N DOCUMENTS(Extrait) ---")
print(filtered_lemma.iloc[:Matrix_size_column, :Matrix_size_line])
print("\n")

print("=== ÉTAPE 5 : TF-IDF (Comparaison) ===")
tfidf_stem = calcul_tfidf(filtered_stem)
tfidf_lemma = calcul_tfidf(filtered_lemma)

print("--- TF-IDF STEMMING (Extrait) ---")
print(tfidf_stem.iloc[:Matrix_size_column, :Matrix_size_line])
print("\n--- TF-IDF LEMMATISATION (Extrait) ---")
print(tfidf_lemma.iloc[:Matrix_size_column, :Matrix_size_line])
print("\n")



print("=== ÉTAPE 6 : VECTORISATION BERT (Sentence-Transformers) ===")

# 1. Préparation des données
# IMPORTANT : BERT a besoin du texte BRUT (phrases complètes avec contexte).
# On n'utilise PAS docs_stem ou docs_lemma, mais 'textes_bruts'.
doc_names = list(textes_bruts.keys())
documents = list(textes_bruts.values())

# 2. Chargement du modèle SBERT
# Le modèle sera téléchargé la première fois (~80MB)
# "all-MiniLM-L6-v2" est un modèle rapide et performant pour la similarité sémantique.
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Calcul des embeddings (Vectorisation)
# normalize_embeddings=True permet d'utiliser le produit scalaire comme similarité cosinus
print("Encodage des documents avec BERT en cours...")
embeddings = model.encode(documents, normalize_embeddings=True)

# 4. Calcul de la matrice de similarité (Cosinus)
similarity_matrix_bert = cosine_similarity(embeddings)

# 5. Création du DataFrame pour affichage
similarity_df_bert = pd.DataFrame(
    similarity_matrix_bert,
    index=doc_names,
    columns=doc_names
)

print("--- SIMILARITÉ BERT (Extrait) ---")
# On utilise tes variables de taille pour contrôler l'affichage
# Note : C'est une matrice Documents x Documents (carrée)
print(similarity_df_bert.iloc[:Matrix_size_line, :Matrix_size_line])
print("\n")

print("=== ÉTAPE 7 : MATRICE DE SIMILARITÉ (Comparaison) ===")
sim_stem = pd.DataFrame(cosine_similarity(tfidf_stem), index=tfidf_stem.index, columns=tfidf_stem.index)
sim_lemma = pd.DataFrame(cosine_similarity(tfidf_lemma), index=tfidf_lemma.index, columns=tfidf_lemma.index)

print("--- SIMILARITÉ STEMMING (5x5) ---")
print(sim_stem.iloc[:Matrix_size_column, :Matrix_size_line])
print("\n--- SIMILARITÉ LEMMATISATION (5x5) ---")
print(sim_lemma.iloc[:Matrix_size_column, :Matrix_size_line])
print("\n")

print("=== ÉTAPE 8 : AFFICHAGE COMPARATIF DES 2 PLOTS ===")

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