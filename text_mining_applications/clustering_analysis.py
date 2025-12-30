import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score

import nltk
from nltk.tokenize import word_tokenize # tokenization based on word
from nltk.stem import SnowballStemmer

# Assure-toi d'avoir les ressources nltk pour découper en phrases
#nltk.download('punkt')
nltk.download('snowball_data')
stemmer = SnowballStemmer("english")

def filter_description_specific(row, token_dict):
    """
    Ne garde que les phrases de la description qui contiennent au moins
    un token du Top-50 spécifique à CETTE université.
    """
    univ_name = row['name'] # Vérifie si la colonne s'appelle 'name' ou 'University' dans ton Parquet
    description = row['description']
    
    # Récupère les tokens spécifiques (Top 50) pour cette fac
    # Si l'université n'est pas dans le JSON (cas rare), on renvoie vide ou tout
    specific_tokens = set(token_dict.get(univ_name, []))
    
    if not specific_tokens or not description:
        return ""
    
    sentences = sent_tokenize(description)
    kept_sentences = []
    
    for sent in sentences:
        sent_lower = sent.lower()
        # La condition magique : on garde si un token SIGNATURE est présent
        if any(token in sent_lower for token in specific_tokens):
            kept_sentences.append(sent)
            
    return " ".join(kept_sentences)

def filter_description_window_stemmed(row, token_dict_stemmed, window_size=5):
    """
    Version hybride : 
    1. Utilise le Stemming pour matcher les mots (ex: 'studying' matche 'studi').
    2. Garde une fenêtre de mots autour du match (pas toute la phrase).
    """
    univ_key = row['name'] # Assure-toi que c'est bien la clé exacte du JSON
    description = str(row['description'])
    
    # Récupération des tokens racines (déjà stemmés dans ton main)
    target_stems = token_dict_stemmed.get(univ_key)
    
    if target_stems is None:
        return "MISSING_KEY"
    if not description:
        return ""
    
    # Tokenisation des mots
    words = word_tokenize(description)
    
    indices_to_keep = set()
    
    # Parcours mot par mot
    for i, w in enumerate(words):
        # On stemme le mot courant du texte pour voir s'il matche un token cible
        w_stem = stemmer.stem(w.lower())
        
        # Le match se fait sur la racine !
        if w_stem in target_stems:
            # Si match, on marque la fenêtre autour
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            for idx in range(start, end):
                indices_to_keep.add(idx)
    
    # Reconstruction
    if not indices_to_keep:
        return ""
    
    sorted_indices = sorted(list(indices_to_keep))
    kept_words = []
    
    last_idx = -1
    for idx in sorted_indices:
        if last_idx != -1 and idx > last_idx + 1:
            kept_words.append("...") 
        kept_words.append(words[idx])
        last_idx = idx
        
    return " ".join(kept_words)

input_desc = "DATA/CLEAN/PARQUET/qs_university_corpus.parquet"
input_file = "DATA/CLEAN/JSON/donnees_traitees_qs.json"

df = pd.read_parquet(input_desc)
df= df.dropna(subset=['description']) # au cas ou mais normalement all good
df['description']= df['description'].astype(str).str.strip()

df=df.head(20).copy()

print(df['description'][0])

with open(input_file, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

univ_to_tokens_stemmed = {}
raw_univ_to_tokens = full_data["tokens"]

for univ, tokens in raw_univ_to_tokens.items():
    # On stocke les racines des tokens pour matcher plus large (study == studies)
    stemmed_tokens = set([stemmer.stem(t) for t in tokens])
    univ_to_tokens_stemmed[univ] = stemmed_tokens


print("Filtrage des descriptions en cours...")

# On s'assure que les noms d'universités correspondent bien (nettoyage basique)
#df['name'] = df['name'].astype(str).str.strip()



df['filtered_desc'] = df.apply(
    lambda row: filter_description_window_stemmed(row, univ_to_tokens_stemmed, window_size=5), axis=1
)


# --- 4. ANALYSE DES RÉSULTATS ---
missing_keys = df[df['filtered_desc'] == "MISSING_KEY"].shape[0]
empty_desc = df[df['filtered_desc'] == ""].shape[0]
success = df[~df['filtered_desc'].isin(["MISSING_KEY", ""])].shape[0]

print(f"\n--- RÉSULTATS ---")
print(f"Universités non trouvées dans le JSON (MISSING_KEY) : {missing_keys}")
print(f"Descriptions vides après filtrage (Aucun mot-clé trouvé) : {empty_desc}")
print(f"Descriptions optimisées avec succès : {success}")

if missing_keys > 0:
    print("\nExemple de noms qui ne matchent pas :")
    print(df[df['filtered_desc'] == "MISSING_KEY"]['name'].head(3).tolist())

# --- 5. EXPORT POUR BERT ---
# On ne garde que les succès
df_final = df[~df['filtered_desc'].isin(["MISSING_KEY", ""])].copy()
texts_optimized = df_final['filtered_desc'].tolist()
print(f"\nListe 'texts_optimized' prête avec {len(texts_optimized)} éléments pour BERT.")

# --- 6. PRÉPARATION POUR BERT ---
# On ne garde que ceux qui ont du texte après filtrage
df_final = df[df['filtered_desc'] != ""].copy()

texts_optimized = df_final['filtered_desc'].tolist()
univ_names_final = df_final['name'].tolist()

print(texts_optimized[0])

print(f"Prêt pour BERT avec {len(texts_optimized)} descriptions optimisées.")

# === 2. EMBEDDINGS BERT SUR LES DESCRIPTIONS ===
model_name = 'all-MiniLM-L6-v2'
#model_name = 'sentence-transformers/all-mpnet-base-v2'
embed_model = SentenceTransformer(model_name)

embeddings = embed_model.encode(texts_optimized, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
embeddings = np.asarray(embeddings)
print("Embeddings shape:", embeddings.shape)

# === 3. CLUSTERING HIÉRARCHIQUE + DENDROGRAMME ===
# Option : sous-échantillonner pour la lisibilité
max_points = 150 # évite d'afficher trop de branche car une branche est une observation
if len(embeddings) > max_points:
    idx_sample = np.random.RandomState(42).choice(len(embeddings), size=max_points, replace=False)
    emb_for_dendro = embeddings[idx_sample]
else:
    emb_for_dendro = embeddings
    

# linkage (complete, average, ward, etc.)
Z = linkage(emb_for_dendro, method='ward')  # ward bien adapté sur embeddings euclidiens[web:81]

plt.figure(figsize=(12, 6))
dendrogram(Z,leaf_rotation=90, leaf_font_size=8, no_labels=True)
plt.title('Dendrogramme hiérarchique (ward) - échantillon')
plt.tight_layout()
#plt.show()

# === 4. MÉTHODE DU COUDE POUR CHOIX DE k ===
inertias = []
sil_scores= []
k_values = range(2, 11)  # range du nombre de cluster

for k in k_values:
    kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans_tmp.fit_predict(embeddings)
    inertias.append(kmeans_tmp.inertia_)

    score = silhouette_score(embeddings, labels, metric ='cosine')
    sil_scores.append(score)
    print(f"Nombre de cluster : {k} et score silhouette : {score}")

print(sil_scores)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o', linestyle="--",color='tab:blue')

plt.xlabel('Nombre de clusters k')
plt.ylabel('Inertie (within-cluster SSE)')
plt.title('Méthode du coude pour KMeans sur embeddings BERT')

xticks=np.arange(2,11,2)
plt.xticks(xticks)

plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()

# Applique le k optimal après méhtode du coude
k_opt = 6
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init='auto')
cluster_ids = kmeans.fit_predict(embeddings)

"""
df_clusters = pd.DataFrame({
    "University": universities,
    "text": texts,
    "cluster_id": cluster_ids
})

# voir docs_tokens not defined


# === 4. THESAURUS ===
thesaurus = {
    'ELITE_RECHERCHE': [
        'research', 'science', 'publication', 'citation', 'ranking', 'world', 
        'leading', 'excellence', 'top', 'theory', 'academic', 'knowledge', 'prestigious'
    ],
    'INNOVATION_TECH': [
        'innovation', 'technology', 'digital', 'future', 'industry', 'new', 
        'entrepreneur', 'develop', 'engineering', 'start', 'modern', 'create', 'tech'
    ],
    'SOCIETE_INCLUSION': [
        'community', 'social', 'public', 'sustainable', 'diversity', 'health', 
        'inclusion', 'access', 'global', 'human', 'environment', 'support', 'civic'
    ],
    'CARRIERE_BUSINESS': [
        'career', 'job', 'business', 'employability', 'skill', 'professional', 
        'work', 'market', 'partner', 'management', 'leader', 'success', 'alumni'
    ]
}

# Pour aller vite, on prépare un set par catégorie
thesaurus_sets = {cat: set(words) for cat, words in thesaurus.items()}

def label_cluster_by_thesaurus(cluster_token_lists, thesaurus_sets):

    
    #cluster_token_lists: liste de listes de tokens (toutes les univ du cluster)
    #Retourne (label, scores) où scores est un dict cat -> count total.
    
    scores = defaultdict(int)
    for tokens in cluster_token_lists:
        for tok in tokens:
            for cat, wordset in thesaurus_sets.items():
                if tok in wordset:
                    scores[cat] += 1

    if not scores:
        return "INDETERMINE", {}

    # tie-breaking explicite
    max_val = max(scores.values())
    best_cats = [c for c, v in scores.items() if v == max_val]

    if max_val == 0:
        label = "INDETERMINE"
    elif len(best_cats) == 1:
        label = best_cats[0]
    else:
        # cluster sémantiquement mixte
        label = "AMBIGU_" + "_".join(sorted(best_cats))

    return label, dict(scores)

cluster_labels = {}
cluster_scores = {}

for cid in sorted(df_cluster['cluster_id'].unique()):
    univ_in_cluster = df_cluster[df_cluster['cluster_id'] == cid]['University'].tolist()
    token_lists = [univ_to_tokens[u] for u in univ_in_cluster]
    label, scores = label_cluster_by_thesaurus(token_lists, thesaurus_sets)
    cluster_labels[cid] = label
    cluster_scores[cid] = scores

# On ajoute le label de cluster à chaque université
df_cluster['cluster_label'] = df_cluster['cluster_id'].map(cluster_labels)

# On ne garde que les clusters bien définis (optionnel)
df_sup = df_cluster[df_cluster['cluster_label'] != "INDETERMINE"].copy()

X = df_sup['text'].values
y = df_sup['cluster_label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('logreg', LogisticRegression(max_iter=1000, n_jobs=-1))
])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))"""