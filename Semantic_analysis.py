import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
from collections import Counter

# --- 1. CHARGEMENT DES DONNÉES ---
print("Chargement des données...")

# A) Les textes nettoyés (depuis le pickle)
with open('DATA-CLEANED/donnees_traitees.pkl', 'rb') as f:
    docs_lemma, _ = pickle.load(f)

# B) Les métadonnées (Région/Continent) depuis le fichier source
# Adaptez le chemin si nécessaire
file_path = 'DATA/PARQUET/qs_university_corpus.parquet' 
df_meta = pd.read_parquet(file_path)

# On crée un dictionnaire de correspondance : { "Nom Université" : "Europe" }
# On s'assure que les noms correspondent (strip pour enlever les espaces)
uni_region_map = dict(zip(df_meta['name'].str.strip(), df_meta['region']))

# --- 2. AGRÉGATION DES MOTS PAR RÉGION ---
print("Agrégation des textes par région...")

# Dictionnaire pour stocker tous les mots : { "General": [...], "Europe": [...], ... }
text_collections = {"GLOBAL (Tout le monde)": []}

for uni_name, tokens in docs_lemma.items():
    # 1. Ajout au pot commun (Général)
    text_collections["GLOBAL (Tout le monde)"].extend(tokens)
    
    # 2. Ajout au pot régional
    # On nettoie le nom pour être sûr de trouver la clé
    clean_name = uni_name.strip()
    
    if clean_name in uni_region_map:
        region = uni_region_map[clean_name]
        
        # Parfois la région est vide ou NaN, on gère ce cas
        if region and isinstance(region, str):
            if region not in text_collections:
                text_collections[region] = []
            text_collections[region].extend(tokens)

print(f"Régions trouvées : {list(text_collections.keys())}")

# --- 3. GÉNÉRATION ET AFFICHAGE DES NUAGES ---
print("Génération des graphiques...")

# Configuration du plot
regions = list(text_collections.keys())
n_regions = len(regions)
# On calcule une taille de grille dynamique (ex: 2 colonnes)
cols = 2
rows = (n_regions + 1) // cols

plt.figure(figsize=(15, 5 * rows))

for i, region in enumerate(regions):
    ax = plt.subplot(rows, cols, i + 1)
    
    # Récupération des mots de la région
    words_list = text_collections[region]
    
    # On compte les fréquences (WordCloud préfère un dictionnaire {mot: freq})
    word_freq = Counter(words_list)
    
    # Création du WordCloud
    # colormap='inferno', 'viridis', 'plasma', 'Pastel1'... changez les couleurs ici
    wc = WordCloud(width=800, height=400, 
                   background_color='white', 
                   max_words=50, 
                   colormap='tab10').generate_from_frequencies(word_freq)
    
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(f"{region} ({len(words_list)} mots)", fontsize=14, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.show()