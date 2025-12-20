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
# Décommentez si nécessaire
# nltk.download('punkt')
# nltk.download('stopwords')

stemmer = nltk.stem.SnowballStemmer("english")
stop_words = list(set(stopwords.words('english'))) + ["'s", "also", "thi", "one", "two"]

def extract_tokens(text):
    """Tokenise, nettoie et racine le texte."""
    if pd.isna(text) or text == "":
        return []
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)
    
    processed_tokens = []
    for t in tokens:
        if t not in string.punctuation and t not in stop_words:
            processed_tokens.append(stemmer.stem(t)) #stemming
    return processed_tokens

def recuperer_documents(fichier, nombre_docs=5): #Par défaut, on prend 5 documents
    """
    Récupère 'nombre_docs' universités.
    Retourne à la fois les tokens traités et le texte brut pour affichage.
    """
    if nombre_docs:
        df = pd.read_csv(fichier, nrows=nombre_docs)
    else:
        df = pd.read_csv(fichier)
        
    docs = {}
    raw_texts = {} # Pour l'affichage de l'étape 1
    
    for _, row in df.iterrows():
        titre = row['title']
        desc = row['QS_description']
        docs[titre] = extract_tokens(desc)
        raw_texts[titre] = desc 
        
    return docs, raw_texts

# --- 2. EXÉCUTION DU PIPELINE COMPLET ---

print("=== ÉTAPE 1 : RÉCUPERATION DU TEXTE BRUT ===")
# Attention, c'est ici que l'on peut choisir le nombre de documents à traiter !!
documents, textes_bruts = recuperer_documents('qs_university_corpus.csv', nombre_docs=5) # endroit où on peut changer le nombre de documents

# Affichage d'un exemple
premier_titre = list(documents.keys())[0]
print(f"Université : {premier_titre}")
print(f"Texte brut (extrait) : {str(textes_bruts[premier_titre])[:100]}...\n")

print("=== ÉTAPE 2 : TOKENISATION & NETTOYAGE ===")
print(f"Tokens générés pour {premier_titre} :")
print(f"{documents[premier_titre][:15]}... (Total : {len(documents[premier_titre])} tokens)\n")

print("=== ÉTAPE 3 : MATRICE INITIALE (Term-Document Matrix) ===")
# Construction du vocabulaire et comptage
vocabulaire = set(token for tokens in documents.values() for token in tokens)
term_frequencies = {doc: Counter(tokens) for doc, tokens in documents.items()}

td_matrix = pd.DataFrame(
    {term: [term_frequencies[doc].get(term, 0) for doc in documents] for term in vocabulaire},
    index=documents.keys()
)
print(f"Dimensions de la matrice : {td_matrix.shape}")
print("Aperçu (5x5) :")
print(td_matrix.iloc[:5, :5])
print("\n")

print("=== ÉTAPE 4 : MATRICE FILTRÉE (Filtre Rareté >= 2) ===")
# On ne garde que les mots présents dans au moins 2 documents
doc_freq_filter = (td_matrix > 0).sum(axis=0)
filtered_td_matrix = td_matrix.loc[:, doc_freq_filter >= 2]

print(f"Dimensions après filtre : {filtered_td_matrix.shape}")
print("Aperçu (5x5) :")
print(filtered_td_matrix.iloc[:5, :5])
print("\n")

print("=== ÉTAPE 5 : TF-IDF ===")
# Calcul TF
row_sums = filtered_td_matrix.sum(axis=1)
row_sums[row_sums == 0] = 1 
tf = filtered_td_matrix.div(row_sums, axis=0)

# Calcul IDF
df_count = (filtered_td_matrix > 0).sum(axis=0)
N = filtered_td_matrix.shape[0]
idf = np.log(N / df_count)

# Matrice Finale TF-IDF
tf_idf = tf.mul(idf, axis=1)

print(f"Dimensions TF-IDF : {tf_idf.shape}")
print("Aperçu (5x5) :")
print(tf_idf.iloc[:5, :5])
print("\n")

print("=== ÉTAPE 6 : MATRICE DE SIMILARITÉ ===")
similarity_matrix_tfidf = cosine_similarity(tf_idf)
similarity_df = pd.DataFrame(similarity_matrix_tfidf, index=tf_idf.index, columns=tf_idf.index)

print(f"Dimensions Similarity : {similarity_df.shape}")
print("Aperçu des scores de similarité (5x5) :")
print(similarity_df.iloc[:5, :5])
print("\n")

print("=== ÉTAPE 7 : AFFICHAGE DU PLOT ===")
def plot_similarity_matrix(similarity_df):
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_df, interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Cosine Similarity')
    plt.title('Document Similarity Matrix')
    
    # Configuration des axes (optionnel)
    plt.xticks(ticks=range(len(similarity_df.columns)), labels=similarity_df.columns, rotation=90)
    plt.yticks(ticks=range(len(similarity_df.index)), labels=similarity_df.index)

    # Boucle pour écrire les valeurs dans chaque case
    for i in range(len(similarity_df)):
        for j in range(len(similarity_df)):
            # On récupère la valeur
            valeur = similarity_df.iloc[i, j]
            
            # On choisit la couleur du texte (blanc si fond foncé, noir si fond clair)
            # 'viridis' est foncé pour les valeurs basses (~0) et clair pour les hautes (~1)
            # Donc on inverse la logique : blanc pour bas, noir pour haut (ou l'inverse selon vos goûts)
            # Pour viridis : jaune (1.0) est clair -> texte noir. violet (0.0) est foncé -> texte blanc.
            couleur_texte = 'black' if valeur > 0.7 else 'white'
            
            plt.text(j, i, f"{valeur:.2f}", 
                     ha='center', va='center', color=couleur_texte, fontsize=8)
    # ---------------------------------------------

    plt.tight_layout()
    plt.show()
plot_similarity_matrix(similarity_df)