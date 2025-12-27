import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
from collections import Counter
import nltk
import networkx as nx
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np

# --- NOUVEAUX IMPORTS POUR LE CLUSTERING AVANCÉ ---
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Téléchargement des ressources NLTK (si pas déjà fait)
# nltk.download('vader_lexicon')
# nltk.download('punkt')

# --- 1. CHARGEMENT DES DONNÉES ---
print("=== 1. CHARGEMENT DES DONNÉES ===")

try:
    with open('DATA-CLEANED/donnees_traitees.pkl', 'rb') as f:
        docs_lemma, td_matrix_lemma = pickle.load(f)
except FileNotFoundError:
    print("Erreur: Fichier 'DATA-CLEANED/donnees_traitees.pkl' introuvable.")
    exit()

# Chargement métadonnées
df_qs = pd.read_parquet('DATA/PARQUET/qs_university_corpus.parquet')
df_the = pd.read_parquet('DATA/PARQUET/the_university_corpus_2021.parquet')

# --- 2. CONFIGURATION TEMPORELLE ---
print("=== 2. CONFIGURATION TEMPORELLE (ODD 2015) ===")

df_qs['period'] = 'Pre-2015 (Avant ODD)'
df_the['period'] = 'Post-2015 (Après ODD)'

meta_combined = pd.concat([df_qs[['name', 'period']], df_the[['name', 'period']]])
meta_combined['name_clean'] = meta_combined['name'].str.strip().str.lower()
uni_period_map = dict(zip(meta_combined['name_clean'], meta_combined['period']))

texts_pre_odd = []
texts_post_odd = []

for name, tokens in docs_lemma.items():
    clean_name = name.strip().lower()
    period = uni_period_map.get(clean_name, "Unknown")
    
    if period == 'Pre-2015 (Avant ODD)':
        texts_pre_odd.extend(tokens)
    elif period == 'Post-2015 (Après ODD)':
        texts_post_odd.extend(tokens)
    else:
        texts_post_odd.extend(tokens)

print(f"Mots Avant ODD : {len(texts_pre_odd)}")
print(f"Mots Après ODD : {len(texts_post_odd)}")

# --- 3. NUAGE DE MOTS TEMPOREL ---
print("\n=== 3. NUAGE DE MOTS : AVANT vs APRÈS ODD ===")

def plot_double_wordcloud(text_list1, title1, text_list2, title2):
    if not text_list1 or not text_list2:
        print("Pas assez de données pour les nuages de mots.")
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    wc1 = WordCloud(width=800, height=400, background_color='white', max_words=50, colormap='winter')
    wc1.generate_from_frequencies(Counter(text_list1))
    axes[0].imshow(wc1, interpolation='bilinear')
    axes[0].set_title(title1, fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    wc2 = WordCloud(width=800, height=400, background_color='white', max_words=50, colormap='autumn')
    wc2.generate_from_frequencies(Counter(text_list2))
    axes[1].imshow(wc2, interpolation='bilinear')
    axes[1].set_title(title2, fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

plot_double_wordcloud(texts_pre_odd, "Focus : AVANT 2015 (Pre-ODD)", 
                      texts_post_odd, "Focus : APRÈS 2015 (Post-ODD)")




# --- 3.5 (CORRIGÉ) : NUAGE DE MOTS GÉOGRAPHIQUE (ESPACÉS) ---
print("\n=== 3.5. ANALYSE SÉMANTIQUE PAR CONTINENT (TOP 15 - ESPACÉS) ===")

col_region = 'region' 

if col_region not in df_qs.columns and col_region not in df_the.columns:
    print(f"ATTENTION : La colonne '{col_region}' est introuvable.")
else:
    # 1. Fusion des données
    dfs_to_concat = []
    if col_region in df_qs.columns: dfs_to_concat.append(df_qs[['name', col_region]])
    if col_region in df_the.columns: dfs_to_concat.append(df_the[['name', col_region]])
    
    meta_geo = pd.concat(dfs_to_concat)
    meta_geo['name_clean'] = meta_geo['name'].str.strip().str.lower()
    uni_region_map = dict(zip(meta_geo['name_clean'], meta_geo[col_region].dropna()))

    # 2. Agrégation
    text_by_region = {}
    for name, tokens in docs_lemma.items():
        clean_name = name.strip().lower()
        region = uni_region_map.get(clean_name, "Inconnu")
        if region != "Inconnu" and isinstance(region, str):
            if region not in text_by_region: text_by_region[region] = []
            text_by_region[region].extend(tokens)

    # 3. Affichage avec ESPACEMENT
    regions_found = sorted(list(text_by_region.keys()))
    
    if len(regions_found) > 0:
        n_regions = len(regions_found)
        ncols = 2
        nrows = (n_regions + 1) // ncols 
        
        # MODIFICATION 1 : On augmente la hauteur par ligne (8 au lieu de 6)
        # figsize = (Largeur, Hauteur)
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 8 * nrows))
        
        # On aplatit le tableau d'axes pour pouvoir itérer facilement (même si grille 2D)
        axes_flat = axes.flatten()

        for i, region in enumerate(regions_found):
            ax = axes_flat[i]
            
            words = text_by_region[region]
            word_counts = Counter(words)
            top_15 = dict(word_counts.most_common(15))
            
            wc = WordCloud(width=800, height=500, # Un peu plus haut
                           background_color='white', 
                           max_words=15, 
                           colormap='tab10',
                           prefer_horizontal=0.9).generate_from_frequencies(top_15)
            
            ax.imshow(wc, interpolation='bilinear')
            
            # MODIFICATION 2 : On ajoute du "padding" (marge) au titre
            ax.set_title(f"RÉGION : {region.upper()}\n(Top 15 mots)", fontsize=18, fontweight='bold', pad=20)
            
            # On ajoute une bordure noire autour pour bien séparer
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
            
            # On enlève juste les ticks (graduations) mais on garde le cadre
            ax.set_xticks([])
            ax.set_yticks([])

        # Masquer les axes vides s'il y en a (ex: 5 régions sur une grille de 6)
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')

        # MODIFICATION 3 : L'espacement magique
        # hspace = espace vertical, wspace = espace horizontal
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        
        plt.show()
    else:
        print("Aucune région trouvée.")



# --- 4. RÉSEAU DE CO-OCCURRENCE ---
print("\n=== 4. RÉSEAU DE CO-OCCURRENCE (Top Mots) ===")

def plot_cooccurrence_network(tokens_list, top_n=30):
    if not tokens_list: return
    
    count = Counter(tokens_list)
    top_words = [word for word, freq in count.most_common(top_n)]
    
    co_occurrences = Counter()
    window_size = 5
    
    for i in range(len(tokens_list) - window_size):
        window = tokens_list[i : i + window_size]
        relevant_words = [w for w in window if w in top_words]
        for j in range(len(relevant_words)):
            for k in range(j + 1, len(relevant_words)):
                w1, w2 = sorted([relevant_words[j], relevant_words[k]])
                if w1 != w2:
                    co_occurrences[(w1, w2)] += 1

    G = nx.Graph()
    for (w1, w2), weight in co_occurrences.items():
        if weight > 5: 
            G.add_edge(w1, w2, weight=weight)
            
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.6, seed=42)
    
    weights = [G[u][v]['weight']/10 for u,v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.3, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    plt.title(f"Réseau sémantique (Top {top_n} mots - Post ODD)", fontsize=15)
    plt.axis('off')
    plt.show()

plot_cooccurrence_network(texts_post_odd)

# --- 5. CONCORDANCE ---
print("\n=== 5. CONCORDANCE (Contexte) ===")
target_words = ["sustainability", "impact", "global", "community"]
if texts_post_odd:
    nltk_text = nltk.Text(texts_post_odd)
    for word in target_words:
        print(f"\n--- Contexte pour : '{word.upper()}' ---")
        try:
            nltk_text.concordance(word, lines=3, width=80)
        except:
            print(f"Mot '{word}' non trouvé.")

# --- 6. SENTIMENT ANALYSIS ---
print("\n=== 6. ANALYSE SÉMANTIQUE (Tonalité) ===")
sia = SentimentIntensityAnalyzer()

def get_sentiment_score(tokens):
    return sia.polarity_scores(" ".join(tokens))['compound']

scores_pre = []
scores_post = []

for name, tokens in docs_lemma.items():
    clean_name = name.strip().lower()
    period = uni_period_map.get(clean_name, "Unknown")
    score = get_sentiment_score(tokens)
    
    if period == 'Pre-2015 (Avant ODD)':
        scores_pre.append(score)
    elif period == 'Post-2015 (Après ODD)':
        scores_post.append(score)

plt.figure(figsize=(8, 6))
plt.boxplot([scores_pre, scores_post], labels=['Avant ODD', 'Après ODD'], patch_artist=True)
plt.title("Évolution de la Tonalité des descriptions", fontsize=14)
plt.ylabel("Positivité (-1 à +1)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ==============================================================================
# --- 7. CLUSTERING AVANCÉ (MÉTHODOLOGIE DU RAPPORT) ---
# ==============================================================================
print("\n=== 7. CLUSTERING AVANCÉ (Méthode du Coude & Comparaison) ===")
print("Objectif : Déterminer le nombre optimal de clusters et comparer K-Means vs Hiérarchique.")

# 1. Préparation des données (Vectorisation)
corpus_list = [" ".join(tokens) for tokens in docs_lemma.values()]
universities_list = list(docs_lemma.keys())

print("Vectorisation TF-IDF en cours...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(corpus_list)


# --- A. MÉTHODE DU COUDE (Elbow Method) ---
print("\n[A] Méthode du Coude (Elbow Method)...")
inertias = []
K_range = range(1, 11)  # Test de 1 à 10 clusters

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-', markersize=8)
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Inertie (Somme des carrés des distances)')
plt.title('Figure 8 : Méthode du Coude')
plt.grid(True)
# On marque visuellement le choix de 5 clusters comme demandé
plt.axvline(x=5, color='r', linestyle='--', label='Choix optimal : k=5') 
plt.legend()
plt.show() 

print(">> D'après la courbe, nous retenons 5 clusters pour la suite.")
n_clusters = 4


# --- B. OPTION 1 : CLUSTERING K-MEANS (k=5) ---
print(f"\n[B] Clustering K-Means (k={n_clusters})...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

# Interprétation des 5 clusters
print("Interprétation sémantique des clusters K-Means :")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
cluster_names_km = {}

for i in range(n_clusters):
    top_terms = [terms[ind] for ind in order_centroids[i, :6]]
    cluster_names_km[i] = f"Groupe {top_terms[0].upper()}"
    print(f" > Cluster {i} (K-Means) : {', '.join(top_terms)}")


# --- C. OPTION 2 : CLUSTERING HIÉRARCHIQUE (k=5) ---
print(f"\n[C] Clustering Hiérarchique (k={n_clusters})...")

# 1. Dendrogramme
plt.figure(figsize=(12, 6))
plt.title("Dendrogramme (Clustering Hiérarchique)")
plt.xlabel("Universités (Échantillon)")
plt.ylabel("Distance (Ward)")
# On calcule la matrice de lien
linkage_matrix = linkage(X.toarray(), method='ward', metric='euclidean')
# On affiche une version tronquée pour la lisibilité
dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=10., show_contracted=True)
plt.show() 

# 2. Calcul des labels
hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
hc_labels = hc.fit_predict(X.toarray())


# --- D. COMPARAISON VISUELLE (PCA 2D) ---
print("\n[D] Visualisation Comparée (K-Means vs Hiérarchique)...")
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X.toarray())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot K-Means
scatter1 = ax1.scatter(coords[:, 0], coords[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.6)
ax1.set_title(f"Partition K-MEANS (k={n_clusters})")
ax1.set_xlabel("Dimension 1")
ax1.set_ylabel("Dimension 2")

# Plot Hiérarchique
scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], c=hc_labels, cmap='plasma', s=50, alpha=0.6)
ax2.set_title(f"Partition HIÉRARCHIQUE (k={n_clusters})")
ax2.set_xlabel("Dimension 1")

plt.tight_layout()
plt.show()

print("\nAnalyse terminée ! Les graphiques ont été générés selon la méthodologie demandée.")