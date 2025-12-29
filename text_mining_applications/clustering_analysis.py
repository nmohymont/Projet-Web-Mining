import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from collections import Counter, defaultdict

input_desc = "DATA/CLEAN/PARQUET/qs_university_corpus.parquet"

df = pd.read_parquet(input_desc)

df= df.dropna(subset=['description']) # au cas ou mais normalement all good
df['description']= df['description'].astype(str).str.strip()

universities = df['name'].tolist()
texts = df['description'].tolist()
print("Nb descriptions:", len(texts))
#print(df.head())

# === 2. EMBEDDINGS BERT SUR LES DESCRIPTIONS ===
#model_name = 'all-MiniLM-L6-v2'
model_name = 'sentence-transformers/all-mpnet-base-v2'
embed_model = SentenceTransformer(model_name)

embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=True)
embeddings = np.asarray(embeddings)
print("Embeddings shape:", embeddings.shape)

# === 3. CLUSTERING HIÉRARCHIQUE + DENDROGRAMME ===
# Option : sous-échantillonner pour la lisibilité
max_points = 150 # évite d'afficher trop de branche car une branche est une observation
if len(embeddings) > max_points:
    idx_sample = np.random.RandomState(42).choice(len(embeddings), size=max_points, replace=False)
    emb_for_dendro = embeddings[idx_sample]
    labels_for_dendro = [universities[i] for i in idx_sample]
else:
    emb_for_dendro = embeddings
    labels_for_dendro = universities

# linkage (complete, average, ward, etc.)
Z = linkage(emb_for_dendro, method='ward')  # ward bien adapté sur embeddings euclidiens[web:81]

plt.figure(figsize=(12, 6))
dendrogram(Z, labels=labels_for_dendro, leaf_rotation=90, leaf_font_size=8, no_labels=True)
plt.title('Dendrogramme hiérarchique (ward) - échantillon')
plt.tight_layout()
#plt.show()

# === 4. MÉTHODE DU COUDE POUR CHOIX DE k ===
inertias = []
k_values = range(2, 11)  # range du nombre de cluster

for k in k_values:
    kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans_tmp.fit(embeddings)
    inertias.append(kmeans_tmp.inertia_)

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

df_clusters = pd.DataFrame({
    "University": universities,
    "text": texts,
    "cluster_id": cluster_ids
})

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
    """
    cluster_token_lists: liste de listes de tokens (toutes les univ du cluster)
    Retourne (label, scores) où scores est un dict cat -> count total.
    """
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