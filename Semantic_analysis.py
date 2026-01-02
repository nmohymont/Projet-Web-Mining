import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
from collections import Counter
import math

import json  # <--- REMPLACE PICKLE
import os

import networkx as nx 
from itertools import combinations

import seaborn as sns

import nltk
from nltk.text import Text

import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from math import pi

#----------------------------
#THINGS TO DO 

# 1 - Mot university encore présent
# 2 - région => Seulement les 3 meilleurs    v  QS et THE
# 3 - 97 Universités seulement => Bizarre     plus valable je pense vu ce que nico a fait
# 4 - The 2021 et 2012 =>Plus utile => UNIQUEMENT ANALYSE TEMPORELLE sans QS          why not mais jsp pk 

# ANALYSE TEMPORELLE - Mots en LIEN avec les ODD
# Il faut que ce soit normalisé !! au nombre d'unif ou a la taille du texte !!  
# 
# Analyse temporelle => The 2025 brider que au top 200         
# Faire en fonction des termes clés des ODD 
# 5 - Cooccurrence => Justifier les liens avec peut-être 10 

#Changer PLK en JSON 
 
# THE 2011,2020,2025 => Top 200 pour analyser les mêmes types d'universités ( pas forcément les mêmes mais les tops de l'époques)
# THE vs QS => Toutes les unif, pas simplement le top 200


#--------------------------------------------------


# 1 - Nuage de mot pour comparer avant/après ODD
# Arrivée ODD en 2015



# --- CONFIGURATION DES FICHIERS ---
# Chemins vers tes 4 fichiers JSON (Adaptez les chemins si besoin)
files_config = {
    'pre_2015': [
        'DATA/CLEAN/JSON/donnees_traitees_the_2012.json'  # Seul fichier avant 2015
    ],
    'post_2015': [
        'DATA/CLEAN/JSON/donnees_traitees_qs.json',       # 2025
        'DATA/CLEAN/JSON/donnees_traitees_the.json',      # 2025
        'DATA/CLEAN/JSON/donnees_traitees_the_2021.json'  # 2021
    ]
}

# --- FONCTION DE CHARGEMENT ---
def load_and_aggregate_tokens(file_list):
    """Charge plusieurs fichiers JSON et combine tous les tokens dans une seule liste."""
    aggregated_tokens = []
    
    for file_path in file_list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 1. Chargement du JSON complet
                data = json.load(f)
                
                # 2. Récupération de la partie "tokens"
                # Rappel de votre structure JSON : { "info": ..., "tokens": {...}, "matrice": ... }
                docs_tokens = data.get("tokens", {})
                
                if not docs_tokens:
                    print(f"   /!\ Avertissement : Aucune donnée 'tokens' trouvée dans {file_path}")
                    continue

                # 3. Agrégation (On ajoute les listes de mots de chaque université)
                count_unis = 0
                for tokens_list in docs_tokens.values():
                    aggregated_tokens.extend(tokens_list)
                    count_unis += 1
                
                print(f"   -> Chargé : {os.path.basename(file_path)} ({count_unis} universités)")
                
        except FileNotFoundError:
            print(f"   /!\ ATTENTION : Fichier introuvable -> {file_path}")
        except json.JSONDecodeError:
            print(f"   /!\ ERREUR JSON : Le fichier {file_path} est malformé ou corrompu.")
        except Exception as e:
            print(f"   /!\ Erreur inattendue sur {file_path} : {e}")
            
    return aggregated_tokens

# --- EXÉCUTION DU CHARGEMENT ---
print("=== CHARGEMENT ET AGRÉGATION DES DONNÉES (JSON) ===")

print("\n1. Traitement du corpus 'AVANT 2015' (Pre-ODD)...")
tokens_pre_2015 = load_and_aggregate_tokens(files_config['pre_2015'])

print("\n2. Traitement du corpus 'APRÈS 2015' (Post-ODD)...")
tokens_post_2015 = load_and_aggregate_tokens(files_config['post_2015'])

print(f"\nTotal mots 'Avant 2015' : {len(tokens_pre_2015)}")
print(f"Total mots 'Après 2015' : {len(tokens_post_2015)}")

# --- GÉNÉRATION DES NUAGES DE MOTS ---
# (Cette partie reste identique car elle travaille sur des listes de mots, peu importe la source)

def plot_compare_wordclouds(tokens1, tokens2, title1, title2):
    if not tokens1 or not tokens2:
        print("Erreur : Pas assez de données pour générer les nuages.")
        return

    # WordCloud prend en entrée une longue chaîne de caractères
    text1 = " ".join(tokens1)
    text2 = " ".join(tokens2)

    # Création des objets WordCloud
    wc1 = WordCloud(width=800, height=400, background_color='white', collocations=False, max_words=50, colormap='winter').generate(text1)
    wc2 = WordCloud(width=800, height=400, background_color='white', collocations=False, max_words=50, colormap='autumn').generate(text2)

    # Affichage
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Nuage 1
    axes[0].imshow(wc1, interpolation='bilinear')
    axes[0].set_title(f"{title1}\n({len(tokens1)} mots analysés)", fontsize=16, fontweight='bold', color='darkblue')
    axes[0].axis('off')

    # Nuage 2
    axes[1].imshow(wc2, interpolation='bilinear')
    axes[1].set_title(f"{title2}\n({len(tokens2)} mots analysés)", fontsize=16, fontweight='bold', color='darkred')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

print("\n=== GÉNÉRATION DES VISUELS ===")
plot_compare_wordclouds(tokens_pre_2015, tokens_post_2015, "AVANT 2015 (Corpus 2012)", "APRÈS 2015 (Corpus 2021-2025)")

# --- AFFICHER LES TOP MOTS ---
print("\n--- TOP 15 MOTS PAR PÉRIODE ---")
counts_pre = Counter(tokens_pre_2015).most_common(15)
counts_post = Counter(tokens_post_2015).most_common(15)

print(f"AVANT 2015 : {counts_pre}")
print(f"APRÈS 2015 : {counts_post}")


# ==============================================================================
# 2 - Nuage de mot en fonction des continent.

# Attention : Difficultés => Les régions/continents ne sont pas dans les fichiers PKL. Il faut faire un lien pour aller les récupérer
# dans les fichiers parquet correspondants.


# --- CONFIGURATION DES SOURCES ---
# On liste ici les paires (Fichier Données Textuelles, Fichier Métadonnées)
# Assure-toi que les noms de fichiers sont corrects
sources = [
    {
        'pkl': 'DATA/CLEAN/PKL/donnees_traitees_qs.pkl', 
        'parquet': 'DATA/CLEAN/PARQUET/qs_university_corpus.parquet',
        'region_col': 'region' # Nom de la colonne région dans ce fichier QS
    },
    {
        'pkl': 'DATA/CLEAN/PKL/donnees_traitees_the.pkl', 
        'parquet': 'DATA/CLEAN/PARQUET/the_university_corpus.parquet',
        'region_col': 'region' # Nom de la colonne région dans ce fichier THE
    },
    {
        'pkl': 'DATA/CLEAN/PKL/donnees_traitees_the_2012.pkl', 
        'parquet': 'DATA/CLEAN/PARQUET/the_university_corpus_2011-2012.parquet',
        'region_col': 'region' # Vérifie si c'est 'region' ou 'continent'
    },
    {
        'pkl': 'DATA/CLEAN/PKL/donnees_traitees_the_2021.pkl', 
        'parquet': 'DATA/CLEAN/PARQUET/the_university_corpus_2021.parquet',
        'region_col': 'region'
    }
]

# Dictionnaire global pour stocker les mots par continent
# Structure : { 'Europe': ['mot1', 'mot2'], 'Asia': ['mot1'] ... }
tokens_by_continent = {}

# ---  TRAITEMENT ET AGREGATION ---
print("=== CHARGEMENT ET FUSION DES DONNÉES ===")

for src in sources:
    pkl_path = src['pkl']
    parquet_path = src['parquet']
    col_region = src['region_col']
    
    try:
        # A. Chargement des Métadonnées (Pour savoir qui est où)
        df_meta = pd.read_parquet(parquet_path)
        
        # On crée un dictionnaire de mapping : Nom Unif -> Région
        # On s'assure que les noms sont propres (minuscules ou strip) pour la correspondance
        # Note : On suppose que la colonne 'name' existe dans le parquet
        mapping_geo = pd.Series(df_meta[col_region].values, index=df_meta['name']).to_dict()
        
        # B. Chargement des Textes Nettoyés
        with open(pkl_path, 'rb') as f:
            docs_lemma, _ = pickle.load(f) # On ignore la matrice, on veut juste les mots
            
        print(f"-> Traitement de {pkl_path}...")
        
        # C. Distribution des mots dans les bons continents
        count_matched = 0
        for uni_name, tokens in docs_lemma.items():
            # On cherche la région de cette université
            # Si le nom exact n'est pas trouvé, on met 'Inconnu'
            region = mapping_geo.get(uni_name, "Inconnu")
            
            # Nettoyage basique du nom de région (ex: gérer les NaN ou vides)
            if region is None or str(region) == 'nan':
                region = "Inconnu"
            
            # Initialisation de la liste si la région est nouvelle
            if region not in tokens_by_continent:
                tokens_by_continent[region] = []
            
            # Ajout des mots
            tokens_by_continent[region].extend(tokens)
            count_matched += 1
            
        print(f"   {count_matched} universités localisées et ajoutées.")

    except FileNotFoundError:
        print(f"   /!\\ Fichier introuvable : {pkl_path} ou {parquet_path}")
    except KeyError as e:
        print(f"   /!\\ Erreur de colonne dans {parquet_path} : {e}. Vérifiez le nom de la colonne région.")

# --- AFFICHAGE DES RÉSULTATS ---
print("\n=== GÉNÉRATION DES NUAGES COMPACTS (TOP 15) ===")

# 1. On récupère TOUS les continents (seuil très bas pour ne rien rater)
# On exclut juste "Inconnu" et les erreurs vides
regions_to_plot = [r for r, t in tokens_by_continent.items() if r != "Inconnu" and len(t) > 10]
regions_to_plot.sort() # On trie par ordre alphabétique pour faire propre

nb_regions = len(regions_to_plot)
print(f"Régions à afficher : {len(regions_to_plot)}")

if nb_regions == 0:
    print("Aucune donnée trouvée.")
else:
    # 2. Configuration de la grille (3 colonnes pour être plus compact)
    cols = 3
    rows = math.ceil(nb_regions / cols)

    # Taille ajustée pour tenir sur un écran standard (Largeur 20, Hauteur proportionnelle)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    
    # Gestion des axes si une seule ligne
    if nb_regions == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, region in enumerate(regions_to_plot):
        tokens = tokens_by_continent[region]
        text = " ".join(tokens)
        
        # 3. Création du WordCloud "Zoomé" (Seulement 15 mots)
        wc = WordCloud(
            width=800,          # Largeur suffisante
            height=500,         # Hauteur standard
            background_color='white', 
            collocations=False, 
            max_words=15,       # <--- LA CONTRAINTE : Seulement 15 mots
            min_font_size=15,   # Texte assez gros
            colormap='Dark2'    # Couleurs foncées pour bien lire
        ).generate(text)
        
        ax = axes[i]
        ax.imshow(wc, interpolation='bilinear')
        
        # Titre propre avec le nombre de mots source
        ax.set_title(f"{region.upper()}\n(Top 15 sur {len(tokens)} mots)", fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
        
        # Ajout d'une bordure fine pour séparer les cases
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#cccccc')

    # 4. Nettoyage des cases vides (s'il y en a)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        axes[j].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.show()

# ==============================================================================
# 3 - Comparaison mots fichier the et fichier qs



#  CONFIGURATION

# Chemins des fichiers
file_matches = 'DATA/CLEAN/CSV/university_mapping_qs_the.csv'
path_pkl_qs = 'DATA/CLEAN/PKL/donnees_traitees_qs.pkl'
path_pkl_the = 'DATA/CLEAN/PKL/donnees_traitees_the.pkl'

# Seuil de matching
SCORE_THRESHOLD = 0.857

# Noms des colonnes du CSV (A adapter si besoin, ex: 'name_x', 'name_y')
COL_QS_NAME = 'QS_Name'   
COL_THE_NAME = 'THE_Name'
COL_SCORE = 'Score'

# CHARGEMENT DES DONNÉES

print("=== 1. CHARGEMENT ===")

# A. Chargement et filtrage du CSV (Qui est le "Juge")
try:
    df_matches = pd.read_csv(file_matches)
    # On ne garde que les lignes avec un bon score
    df_filtered = df_matches[df_matches[COL_SCORE] > SCORE_THRESHOLD]
    print(f"-> CSV chargé. Paires valides (> {SCORE_THRESHOLD}) : {len(df_filtered)}")
except FileNotFoundError:
    print(f"Erreur : Impossible de trouver {file_matches}")
    exit()

# B. Chargement des PKL (Les données textuelles propres)
def load_pkl_dict(path):
    try:
        with open(path, 'rb') as f:
            # Le pkl contient un tuple (dictionnaire, matrice), on prend le dictionnaire [0]
            data = pickle.load(f)
            return data[0] 
    except FileNotFoundError:
        print(f"Erreur : Impossible de trouver {path}")
        return {}

print("-> Chargement du PKL QS...")
docs_qs = load_pkl_dict(path_pkl_qs)

print("-> Chargement du PKL THE...")
docs_the = load_pkl_dict(path_pkl_the)


# FILTRAGE ET AGREGATION
print("\n=== 2. CROISEMENT DES DONNÉES ===")

tokens_qs_final = []
tokens_the_final = []
count_match = 0

# On parcourt le CSV filtré
for index, row in df_filtered.iterrows():
    name_qs = row[COL_QS_NAME]
    name_the = row[COL_THE_NAME]
    
    # On vérifie si l'université existe bien dans nos fichiers PKL
    if name_qs in docs_qs and name_the in docs_the:
        # On ajoute les mots de cette université à la liste globale
        tokens_qs_final.extend(docs_qs[name_qs])
        tokens_the_final.extend(docs_the[name_the])
        count_match += 1

print(f"-> Analyse basée sur {count_match} universités communes.")
print(f"-> Total mots QS  : {len(tokens_qs_final)}")
print(f"-> Total mots THE : {len(tokens_the_final)}")

# VISUALISATION
print("\n=== 3. GÉNÉRATION DES NUAGES DE MOTS ===")

# Fonction pour créer et afficher
def plot_clouds(tokens1, tokens2):
    # Transformation liste -> texte
    text1 = " ".join(tokens1)
    text2 = " ".join(tokens2)
    
    # Création des WordClouds
    # Note : Comme les PKL sont déjà filtrés (pas de 'university', pas de 'the'...), 
    # on n'a pas besoin de remettre beaucoup de stopwords ici.
    wc_qs = WordCloud(width=800, height=400, background_color='white', 
                      colormap='Blues', collocations=False, max_words=50).generate(text1)
                      
    wc_the = WordCloud(width=800, height=400, background_color='white', 
                       colormap='Reds', collocations=False, max_words=50).generate(text2)

    # Affichage
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    axes[0].imshow(wc_qs, interpolation='bilinear')
    axes[0].set_title(f"Vocabulaire QS\n(Sur les {count_match} universités matchées)", fontsize=16, color='darkblue')
    axes[0].axis('off')
    
    axes[1].imshow(wc_the, interpolation='bilinear')
    axes[1].set_title(f"Vocabulaire THE\n(Sur les {count_match} universités matchées)", fontsize=16, color='darkred')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Lancement
if count_match > 0:
    plot_clouds(tokens_qs_final, tokens_the_final)
else:
    print("Aucune correspondance trouvée entre le CSV et les fichiers PKL. Vérifiez les noms.")


# =============================================================================
# 4 - Graphe de co-occurrence combiné (4 fichiers)


# CONFIGURATION

# Liste des 4 fichiers à combiner
files_to_combine = [
    'DATA/CLEAN/PKL/donnees_traitees_qs.pkl',
    'DATA/CLEAN/PKL/donnees_traitees_the.pkl',
    'DATA/CLEAN/PKL/donnees_traitees_the_2021.pkl',
    'DATA/CLEAN/PKL/donnees_traitees_the_2012.pkl'
]

# --- PARAMÈTRES DU GRAPHE ---
# Nombre de mots les plus fréquents à afficher.
# CONSEIL : Avec 4 fichiers, gardez ce chiffre entre 50 et 100 pour que ce soit lisible.
# Si vous mettez 520, le graphique sera illisible (trop de noeuds).
TOP_N_WORDS = 60

# Seuil minimum de cooccurrence
# Puisqu'on combine 4 fichiers, on augmente un peu ce seuil pour ne garder que les liens forts.
MIN_EDGE_WEIGHT = 10 

# CHARGEMENT ET AGREGATION

print("=== 1. CHARGEMENT ET FUSION DES 4 CORPUS ===")

all_docs_list = [] # Liste qui contiendra les listes de mots de TOUTES les universités combinées

for file_path in files_to_combine:
    try:
        with open(file_path, 'rb') as f:
            # On récupère le dictionnaire (élément 0 du tuple sauvegardé)
            data = pickle.load(f)
            docs_lemma = data[0]
            
            # On ajoute les listes de tokens de ce fichier à la liste globale
            # docs_lemma.values() est une liste de listes de mots [['mot1', 'mot2'], ['mot3']...]
            for tokens in docs_lemma.values():
                all_docs_list.append(tokens)
                
        print(f"-> Chargé avec succès : {file_path} ({len(docs_lemma)} universités)")
        
    except FileNotFoundError:
        print(f"/!\ Erreur : Fichier introuvable -> {file_path}")
    except Exception as e:
        print(f"/!\ Erreur sur {file_path} : {e}")

print(f"\n-> TOTAL DOCUMENTS ANALYSÉS : {len(all_docs_list)}")

# CALCULS STATISTIQUES
print("\n=== 2. ANALYSE DES FRÉQUENCES ET COOCCURRENCES ===")

# A. Sélection des Top Mots sur l'ensemble combiné
# On aplatit la liste de listes pour compter tous les mots
all_tokens_flat = [token for doc in all_docs_list for token in doc]
word_counts = Counter(all_tokens_flat)

# On récupère les N mots les plus fréquents
top_words_dict = dict(word_counts.most_common(TOP_N_WORDS))
top_words_set = set(top_words_dict.keys())

print(f"-> Top {TOP_N_WORDS} mots sélectionnés (ex: {list(top_words_dict.keys())[:5]}...)")

# B. Calcul des Cooccurrences
co_occurrence_counts = Counter()

# On parcourt chaque document de notre liste géante
for tokens in all_docs_list:
    # 1. On ne garde que les mots du Top N
    filtered_tokens = [t for t in tokens if t in top_words_set]
    
    # 2. On prend les mots uniques (pour éviter de compter 2 fois si le mot est répété dans le même texte)
    unique_tokens = sorted(list(set(filtered_tokens)))
    
    # 3. On génère les paires
    if len(unique_tokens) > 1:
        pairs = list(combinations(unique_tokens, 2))
        co_occurrence_counts.update(pairs)

print(f"-> Liens calculés. Total de paires uniques trouvées : {len(co_occurrence_counts)}")

# CONSTRUCTION DU GRAPHE (NetworkX)
print("\n=== 3. GÉNÉRATION DU GRAPHE COMBINÉ ===")

G = nx.Graph()

# A. Ajout des Nœuds (Mots)
for word, count in top_words_dict.items():
    # On divise la taille par un facteur pour éviter d'avoir des boules géantes (car on a 4x plus de données)
    G.add_node(word, size=count)

# B. Ajout des Liens (Cooccurrences)
edges_added = 0
for pair, weight in co_occurrence_counts.items():
    if weight >= MIN_EDGE_WEIGHT:
        G.add_edge(pair[0], pair[1], weight=weight)
        edges_added += 1

print(f"-> Graphe final : {G.number_of_nodes()} nœuds, {edges_added} liens (filtre poids >= {MIN_EDGE_WEIGHT}).")

# VISUALISATION

plt.figure(figsize=(18, 14)) # Taille augmentée pour la lisibilité

# 1. Disposition (Spring Layout)
# k=0.6 : on écarte un peu plus les nœuds car il y a beaucoup de liens
pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42)

# 2. Gestion des tailles (Normalisation)
# Comme on a beaucoup de données, on ajuste le facteur de taille pour que ce soit joli
base_size = [G.nodes[n]['size'] for n in G.nodes]
max_size = max(base_size) if base_size else 1
node_sizes = [(s / max_size) * 3000 for s in base_size] # Taille max de 3000

# Épaisseur des traits
max_weight = max([G.edges[u, v]['weight'] for u, v in G.edges]) if G.edges else 1
edge_widths = [(G.edges[u, v]['weight'] / max_weight) * 5 for u, v in G.edges] # Épaisseur max 5

# 3. Dessin
# Nœuds
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#69b3a2', alpha=0.9, edgecolors='white')

# Labels
nx.draw_networkx_labels(G, pos, font_size=11, font_family='sans-serif', font_weight='bold')

# Liens
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray')

plt.title(f"Réseau de Cooccurrence Combiné (4 Sources) - Top {TOP_N_WORDS} mots", fontsize=20)
plt.axis('off')

# Ajout d'une petite légende pour expliquer
plt.figtext(0.5, 0.02, 
            f"Basé sur {len(all_docs_list)} descriptions d'universités (QS 25, THE 25, THE 21, THE 12).\n"
            f"Lien affiché si les mots apparaissent ensemble dans au moins {MIN_EDGE_WEIGHT} textes.", 
            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

plt.show()


# ==============================================================================
# 5 - Analyse de concordance 

# CONFIGURATION
# 1. Fichiers NETTOYÉS (JSON) -> Pour voir comment l'algo "voit" les mots (Partie A)
files_json = [
    'DATA/CLEAN/JSON/donnees_traitees_qs.json', 
    'DATA/CLEAN/JSON/donnees_traitees_the.json'
]

# 2. Fichiers BRUTS (Parquet) -> Pour lire les vraies phrases complètes (Partie B)
file_qs_raw = 'DATA/CLEAN/PARQUET/qs_university_corpus.parquet'
file_the_raw = 'DATA/CLEAN/PARQUET/the_university_corpus.parquet'

# Tes mots cibles
TARGET_WORDS = ["sustainable", "impact", "innovation"]

# -----------------------------------------------------------------------------
# PARTIE A : CONCORDANCE TECHNIQUE (KWIC - Key Word In Context)
# Objectif : Voir les voisins immédiats des mots lemmatisés
# -----------------------------------------------------------------------------
print("\n" + "="*50)
print("PARTIE A : CONCORDANCE VISUELLE (Sur tokens nettoyés)")
print("="*50)

all_tokens_clean = []

# Chargement des JSON
for f_path in files_json:
    try:
        with open(f_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # On récupère la sous-partie "tokens"
            # Structure JSON : { "info": ..., "tokens": { "Univ": ["mot1"] } }
            docs_tokens = data.get('tokens', {})
            
            # On agglomère tous les mots de toutes les universités
            for tokens in docs_tokens.values():
                all_tokens_clean.extend(tokens)
                
        print(f"-> Chargé pour analyse technique : {os.path.basename(f_path)}")
        
    except FileNotFoundError:
        print(f"Erreur : Fichier introuvable -> {f_path}")

# Création de l'objet NLTK
# Cela permet d'utiliser la fonction .concordance() très pratique
text_object = Text(all_tokens_clean)

for word in TARGET_WORDS:
    print(f"\n--- Scan du mot : '{word}' (Contexte technique) ---")
    # Affiche le mot centré avec ses voisins (avant/après)
    # width=90 définit la largeur de l'affichage
    text_object.concordance(word, lines=5, width=90)


# -----------------------------------------------------------------------------
# PARTIE B : EXTRACTION DE SENS (Pour le Rapport)
# Objectif : Trouver des phrases réelles et lisibles pour citer en exemple
# -----------------------------------------------------------------------------
print("\n" + "="*50)
print("PARTIE B : DÉFINITIONS ET PHRASES COMPLÈTES (Sur texte brut)")
print("="*50)

try:
    # Chargement des vraies phrases (Parquet)
    df_qs = pd.read_parquet(file_qs_raw)
    df_the = pd.read_parquet(file_the_raw)
    
    # On combine les descriptions brutes
    # On s'assure que tout est converti en string pour éviter les erreurs
    all_descriptions_raw = list(df_qs['description'].astype(str)) + list(df_the['description'].astype(str))
    
    print(f"-> Corpus brut chargé : {len(all_descriptions_raw)} descriptions.")

    for target in TARGET_WORDS:
        print(f"\n>>> CONTEXTE RÉEL POUR : '{target.upper()}'")
        
        found_sentences = []
        
        for desc in all_descriptions_raw:
            # On utilise le tokenizer de phrases de NLTK
            # Il est plus malin qu'un simple split('.')
            sentences = nltk.sent_tokenize(desc)
            
            for sent in sentences:
                # Recherche simple (insensible à la casse)
                if target in sent.lower():
                    # Nettoyage des sauts de ligne
                    clean_sent = sent.replace('\n', ' ').strip()
                    found_sentences.append(clean_sent)
        
        # Filtrage : On garde les phrases "intéressantes" (assez longues)
        # > 60 caractères évite les titres comme "Sustainable Innovation."
        long_sentences = [s for s in found_sentences if len(s) > 60]
        
        if long_sentences:
            # On affiche jusqu'à 3 exemples
            for i, s in enumerate(long_sentences[:3]):
                print(f"📖 Ex {i+1}: {s}\n")
        else:
            print("Aucune phrase significative trouvée.")

except Exception as e:
    print(f"Erreur lors de la lecture des fichiers Parquet : {e}")

print("\nAnalyse terminée.")


# =============================================================================
# 6 - N-gram 


# ==============================================================================
# 6 - N-gram (Version JSON : Séquences de Mots-Clés)
# ==============================================================================

# CONFIGURATION
# On utilise les fichiers JSON (Mots nettoyés)
files_json = [
    'DATA/CLEAN/JSON/donnees_traitees_qs.json',
    'DATA/CLEAN/JSON/donnees_traitees_the.json',
    'DATA/CLEAN/JSON/donnees_traitees_the_2021.json'
]

# L'expression déclencheur
# ATTENTION : Comme on est en JSON nettoyé, "is", "the", "of" n'existent plus !
# Il faut chercher un concept.
# Exemples valides : "research", "sustainable", "campus", "global"
TRIGGER_WORD = "research" 

# La longueur de la suite
# 5 ou 6 suffisent pour des tokens (car 14 mots-clés, c'est énorme sans les mots de liaison)
N_GRAM_SIZE = 6

# CHARGEMENT ET PRÉPARATION
print("=== CHARGEMENT DES SÉQUENCES DE MOTS-CLÉS ===")

all_tokens_lists = []

for f_path in files_json:
    try:
        with open(f_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # On récupère les listes de mots de chaque université
            docs_tokens = data.get('tokens', {})
            
            # On ajoute chaque liste de mots à notre corpus
            for tokens in docs_tokens.values():
                if tokens: # Si la liste n'est pas vide
                    all_tokens_lists.append(tokens)
                    
        print(f"-> Chargé : {os.path.basename(f_path)}")
        
    except FileNotFoundError:
        print(f"/!\\ Fichier introuvable : {f_path}")

print(f"Analyse sur {len(all_tokens_lists)} universités.")


# ALGORITHME D'EXTRACTION DE N-GRAMMES
print(f"\nRecherche des {N_GRAM_SIZE} mots-clés suivant le terme : '{TRIGGER_WORD}'...")

found_sequences = []
trigger_token = TRIGGER_WORD.lower().strip()

# On parcourt chaque université (chaque liste de tokens)
for tokens in all_tokens_lists:
    # On cherche le mot déclencheur dans la liste
    # Note : Un mot peut apparaître plusieurs fois dans une description
    for i in range(len(tokens) - 1 - N_GRAM_SIZE):
        
        # Si le mot courant est notre déclencheur
        if tokens[i] == trigger_token:
            # On capture les N mots qui suivent
            # C'est une extraction de liste, pas de chaîne de caractères
            sequence_list = tokens[i + 1 : i + 1 + N_GRAM_SIZE]
            
            # On rejoint en string pour pouvoir les compter facilement
            sequence_str = " ".join(sequence_list)
            found_sequences.append(sequence_str)

# RÉSULTATS ET AFFICHAGE

counts = Counter(found_sequences)
top_15 = counts.most_common(15)

print(f"\n--- TOP 15 SÉQUENCES DE THÈMES APRÈS '{TRIGGER_WORD.upper()}' ---\n")

if not top_15:
    print(f"Aucune séquence trouvée pour '{TRIGGER_WORD}'. Essayez un mot plus fréquent (ex: 'student', 'campus').")
else:
    for i, (seq, count) in enumerate(top_15):
        # On affiche une barre de chargement visuelle pour la fréquence
        bar = "|" * (count // 2) if count > 1 else "|"
        print(f"{i+1:02d}. [{count:3d} x] {trigger_token.upper()} -> {seq}")

# Note explicative pour l'utilisateur
print("\n" + "-"*60)
print("NOTE D'INTERPRÉTATION :")
print("Ceci analyse les 'mots-clés'. La grammaire est cassée (pas de 'is', 'the').")
print(f"Exemple : '{TRIGGER_WORD} center excellence' signifie probablement '...research center of excellence...'")
print("-"*60)


# ==============================================================================
# 7 - Evolution temporelle des termes 



# 1. CONFIGURATION

# Liste temporelle des fichiers
# On essaie de garder la même source (THE) pour la cohérence temporelle
# On ajoute QS 2025 à la fin pour comparer la vision actuelle
files_timeline = [
    {'year': '2012', 'source': 'THE', 'path': 'DATA/CLEAN/PKL/donnees_traitees_the_2012.pkl'},
    {'year': '2021', 'source': 'THE', 'path': 'DATA/CLEAN/PKL/donnees_traitees_the_2021.pkl'},
    {'year': '2025', 'source': 'THE', 'path': 'DATA/CLEAN/PKL/donnees_traitees_the.pkl'},
    {'year': '2025', 'source': 'QS',  'path': 'DATA/CLEAN/PKL/donnees_traitees_qs.pkl'}
]

# THEMES À ANALYSER

# Thème 1 : L'impact et le "Green" (Le plus probable d'augmenter)
KEYWORDS = ["sustainable", "impact", "global", "community", "environment"]

# Thème 2 : La Tech et l'Innovation
# KEYWORDS = ["digital", "technology", "innovation", "online", "data"]

# Thème 3 : Les Fondamentaux (Devraient être stables ou baisser légèrement)
# KEYWORDS = ["research", "teaching", "library", "campus", "student"]

# CALCUL DES FRÉQUENCES RELATIVES
print("=== CALCUL DE L'ÉVOLUTION TEMPORELLE ===")

results = []

for item in files_timeline:
    label = f"{item['year']} ({item['source']})"
    path = item['path']
    
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            docs_lemma = data[0] # Le dictionnaire est le premier élément
            
            # 1. On rassemble tous les mots de cette année-là
            all_tokens = [t for doc in docs_lemma.values() for t in doc]
            total_words_count = len(all_tokens)
            
            # 2. On compte les occurrences
            counts = Counter(all_tokens)
            
            # 3. Pour chaque mot-clé, on calcule son %
            row = {'Year_Label': label, 'Year_Int': int(item['year'])}
            
            for word in KEYWORDS:
                # Fréquence relative (pour 10 000 mots pour que ce soit lisible)
                if total_words_count > 0:
                    freq = (counts[word] / total_words_count) * 10000
                else:
                    freq = 0
                row[word] = freq
            
            results.append(row)
            print(f"-> Traité : {label} (Base : {total_words_count} mots)")

    except FileNotFoundError:
        print(f"ERREUR : Fichier introuvable {path}")

# Création du DataFrame
df_trends = pd.DataFrame(results)

# VISUALISATION (LINE CHART)
print("\n=== GÉNÉRATION DU GRAPHIQUE ===")

# Configuration du style
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# On trace une ligne pour chaque mot
# On utilise les marqueurs 'o' pour bien voir les points de données
for word in KEYWORDS:
    sns.lineplot(data=df_trends, x='Year_Label', y=word, marker='o', linewidth=2.5, label=word)

plt.title(f"Évolution temporelle des termes clés (Fréquence pour 10 000 mots)", fontsize=16, fontweight='bold')
plt.ylabel("Fréquence (pour 10k mots)", fontsize=12)
plt.xlabel("Année / Source", fontsize=12)

# Légende
plt.legend(title="Mots Clés", title_fontsize='12', fontsize='11', loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# TABLEAU DE DONNÉES (POUR LE RAPPORT)
print("\n--- TABLEAU DES VALEURS ---")
# On met l'année en index pour l'affichage
print(df_trends.set_index('Year_Label')[KEYWORDS].round(2))



#-------------------------
#Semantic_analysis




'''


# ==============================================================================
# ANALYSE SÉMANTIQUE 
#=============================================================================

print("\n=== ANALYSE SÉMANTIQUE DES DESCRIPTIONS D'UNIVERSITÉS ===")

# -----------------------------------------------
# 1 - thésaurus de sentiment 




# DÉFINITION DU THÉSAURUS (LES DIMENSIONS)

# Tu peux enrichir ces listes selon ce que tu as vu dans les nuages de mots
thesaurus = {
    'EXCELLENCE / PRESTIGE': [
        'leading', 'world', 'top', 'rank', 'excellence', 'best', 'prestigous', 
        'elite', 'reputation', 'quality', 'leader', 'award', 'prize', 'noble'
    ],
    'INNOVATION / FUTUR': [
        'innovation', 'technology', 'digital', 'future', 'modern', 'new', 
        'creative', 'develop', 'tech', 'smart', 'entrepreneur', 'start'
    ],
    'SOCIÉTAL / INCLUSION': [
        'community', 'social', 'public', 'sustainable', 'environment', 'health', 
        'diversity', 'inclusion', 'access', 'global', 'impact', 'human', 'green'
    ],
    'CARRIÈRE / EMPLOI': [
        'career', 'job', 'employability', 'industry', 'business', 'skill', 
        'professional', 'work', 'alumni', 'salary', 'market', 'partner'
    ],
    'RECHERCHE / ACADÉMIQUE': [
        'research', 'study', 'science', 'knowledge', 'academic', 'theory', 
        'publication', 'discover', 'institute', 'phd', 'faculty', 'professor'
    ]
}

# CHARGEMENT DES DONNÉES
# On compare les deux visions actuelles (QS vs THE 2025) et le passé (2012)
files_to_compare = {
    'THE 2012 (Passé)': 'DATA/CLEAN/PKL/donnees_traitees_the_2012.pkl',
    'THE 2021 (Récent)': 'DATA/CLEAN/PKL/donnees_traitees_the_2021.pkl',
    'THE 2025 (Académique)': 'DATA/CLEAN/PKL/donnees_traitees_the.pkl',
    'QS 2025 (Marketing)':  'DATA/CLEAN/PKL/donnees_traitees_qs.pkl'
}

scores_global = {}

print("=== CALCUL DES SCORES DE TONALITÉ ===")

for label, path in files_to_compare.items():
    try:
        with open(path, 'rb') as f:
            docs_lemma, _ = pickle.load(f)
            
            # Initialisation des compteurs pour ce fichier
            cat_counts = {cat: 0 for cat in thesaurus.keys()}
            total_words = 0
            
            # On parcourt tous les mots de toutes les universités
            for tokens in docs_lemma.values():
                total_words += len(tokens)
                for token in tokens:
                    # On vérifie si le mot appartient à une catégorie
                    for category, keywords in thesaurus.items():
                        if token in keywords:
                            cat_counts[category] += 1
            
            # Normalisation : Score pour 10 000 mots (pour comparer des fichiers de tailles différentes)
            # On évite la division par zéro
            if total_words > 0:
                normalized_scores = {k: (v / total_words) * 10000 for k, v in cat_counts.items()}
            else:
                normalized_scores = {k: 0 for k in cat_counts.items()}
                
            scores_global[label] = normalized_scores
            print(f"-> Traité : {label}")

    except FileNotFoundError:
        print(f"Erreur : {path} introuvable.")

# VISUALISATION : RADAR CHART (SPIDER PLOT)
# C'est la partie un peu technique pour faire un beau graphique en toile d'araignée

print("\n=== GÉNÉRATION DU RADAR CHART ===")

# Préparation des données pour le plot
categories = list(thesaurus.keys())
N = len(categories)

# Calcul des angles pour le cercle
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1] # Fermer la boucle

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

# Définir l'axe (le sens des aiguilles d'une montre)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Dessiner les axes par catégorie
plt.xticks(angles[:-1], categories, size=11, fontweight='bold')

# Dessiner les labels de l'axe Y
ax.set_rlabel_position(0)
plt.yticks([50, 100, 150, 200], ["50", "100", "150", "200"], color="grey", size=7)
plt.ylim(0, 250) # Adapter selon tes scores max (si tes courbes sortent, augmente 250)

# Ajouter les courbes pour chaque fichier
colors = ['orange', 'red', 'blue','green'] # 2012, THE, QS

for i, (label, scores) in enumerate(scores_global.items()):
    values = [scores[cat] for cat in categories]
    values += values[:1] # Fermer la boucle
    
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=label, color=colors[i])
    ax.fill(angles, values, color=colors[i], alpha=0.1)

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title("Profil Tonal des Universités : Comparaison Multi-Sources", size=16, y=1.1)
plt.show()

# BONUS : "ALLER PLUS LOIN" (TERMES DISTINCTIFS)

# Ici, on ne cherche pas des catégories prédéfinies.
# On demande au code : "Quels mots sont UNIQUES à QS par rapport à THE ?"

print("\n=== ALLER PLUS LOIN : LES TERMES DISTINCTIFS ===")
print("Quels mots sont sur-représentés chez QS (Marketing) vs THE (Académique) ?")

# On charge QS et THE 2025 pour comparer
def get_tokens(path):
    with open(path, 'rb') as f:
        return [t for doc in pickle.load(f)[0].values() for t in doc]

tokens_qs = get_tokens(files_to_compare['QS 2025 (Marketing)'])
tokens_the = get_tokens(files_to_compare['THE 2025 (Académique)'])

# Compte
counts_qs = pd.Series(tokens_qs).value_counts(normalize=True) # Fréquence relative
counts_the = pd.Series(tokens_the).value_counts(normalize=True)

# Création d'un DataFrame comparatif
df_comp = pd.DataFrame({'QS': counts_qs, 'THE': counts_the}).fillna(0)

# On garde seulement les mots assez fréquents (pour éviter le bruit)
df_comp = df_comp[(df_comp['QS'] > 0.0001) & (df_comp['THE'] > 0.0001)]

# Calcul du Ratio : (Fréquence QS / Fréquence THE)
# Si Ratio > 1 : Mot "QS". Si Ratio < 1 : Mot "THE".
df_comp['Ratio_QS_vs_THE'] = df_comp['QS'] / df_comp['THE']

print("\n--- TOP 10 MOTS 'MARQUEUR QS' (Ratio le plus élevé) ---")
print(df_comp.sort_values('Ratio_QS_vs_THE', ascending=False).head(10)[['Ratio_QS_vs_THE']])

print("\n--- TOP 10 MOTS 'MARQUEUR THE' (Ratio le plus faible) ---")
print(df_comp.sort_values('Ratio_QS_vs_THE', ascending=True).head(10)[['Ratio_QS_vs_THE']])





# ==============================================================================
# Machine Learning : Classification supervisée
#=============================================================================

print("\n=== ANALYSE SÉMANTIQUE DES DESCRIPTIONS D'UNIVERSITÉS (Pré/Post ODD) ===")


# 1 - Une méthode de classification (supervisée)

#PARTIE 1 - Création et rajout des labels 

# CONFIGURATION
# Fichier d'entrée (QS ou THE)
input_file = 'DATA/CLEAN/PKL/donnees_traitees_qs.pkl'

# Fichier de sortie (Le nouveau fichier que tu demandes)
output_csv = 'DATA/CLEAN/CSV/MACHINE-LEARNING/universites_qs_label.csv'

# Ton Thésaurus (Les règles du jeu)
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

#  CALCUL ET CRÉATION DU FICHIER
print("=== ANALYSE ET CLASSIFICATION DES UNIVERSITÉS ===")

data_rows = []

try:
    with open(input_file, 'rb') as f:
        data = pickle.load(f)[0]
        
        print(f"-> Lecture de {len(data)} universités...")

        for name, tokens in data.items():
            # 1. Initialiser les scores à 0
            scores = {cat: 0 for cat in thesaurus.keys()}
            
            # 2. Compter les mots-clés présents
            for token in tokens:
                for cat, keywords in thesaurus.items():
                    if token in keywords:
                        scores[cat] += 1
            
            # 3. Trouver la catégorie dominante
            best_cat = max(scores, key=scores.get)
            max_val = scores[best_cat]
            
            # 4. On enregistre le résultat (même si le score est faible, on le garde pour info)
            # On ajoute aussi le score pour pouvoir filtrer plus tard si besoin
            data_rows.append({
                'University': name,
                'Dominant_Value': best_cat,
                'Score_Intensity': max_val,
                'Full_Scores': str(scores) # On garde le détail au cas où
            })

except FileNotFoundError:
    print(f"Erreur : Impossible de trouver {input_file}")
    exit()

# Création du DataFrame
df_results = pd.DataFrame(data_rows)

# SAUVEGARDE

# On s'assure que le dossier existe
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Sauvegarde
df_results.to_csv(output_csv, index=False)

print(f"\nSUCCÈS ! Fichier créé : {output_csv}")
print(f"-> Total lignes : {len(df_results)}")

# Aperçu
print("\n--- Aperçu des 5 premières lignes ---")
print(df_results.head())

print("\n--- Répartition des Valeurs ---")
print(df_results['Dominant_Value'].value_counts())




# Partie 2 - Apprentissage supervisé sur QS et test sur THE

#  CONFIGURATION ET THÉSAURUS
file_qs = 'DATA/CLEAN/PKL/donnees_traitees_qs.pkl'  # Fichier d'ENTRAÎNEMENT
file_the = 'DATA/CLEAN/PKL/donnees_traitees_the.pkl' # Fichier de TEST

# Le Thésaurus (La "Vérité" commune pour étiqueter les deux corpus)
thesaurus = {
    'ELITE_RECHERCHE': ['research', 'science', 'publication', 'citation', 'ranking', 'world', 'leading', 'excellence', 'top', 'theory', 'academic', 'knowledge'],
    'INNOVATION_TECH': ['innovation', 'technology', 'digital', 'future', 'industry', 'new', 'entrepreneur', 'develop', 'engineering', 'start', 'tech'],
    'SOCIETE_INCLUSION': ['community', 'social', 'public', 'sustainable', 'diversity', 'health', 'inclusion', 'access', 'global', 'human', 'environment'],
    'CARRIERE_BUSINESS': ['career', 'job', 'business', 'employability', 'skill', 'professional', 'work', 'market', 'partner', 'management', 'leader']
}

# Fonction pour charger et étiqueter automatiquement un fichier
def load_and_label(file_path, source_name):
    texts = []
    labels = []
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)[0]
            print(f"-> Chargement de {source_name}...")
            
            for name, tokens in data.items():
                # Calcul des scores
                scores = {cat: 0 for cat in thesaurus.keys()}
                for token in tokens:
                    for cat, keywords in thesaurus.items():
                        if token in keywords:
                            scores[cat] += 1
                
                # On prend la catégorie dominante
                best_cat = max(scores, key=scores.get)
                
                # On ne garde que si le signal est assez fort (Score > 2)
                if scores[best_cat] > 2:
                    texts.append(" ".join(tokens))
                    labels.append(best_cat)
                    
        print(f"   {len(texts)} universités qualifiées dans {source_name}.")
        return texts, labels
    except FileNotFoundError:
        print(f"Erreur : {file_path} introuvable.")
        return [], []

# PRÉPARATION DES DONNÉES (TRAIN VS TEST)
print("=== 1. CRÉATION DES DATASETS ===")

# ENTRAÎNEMENT : 100% de QS
X_train_text, y_train = load_and_label(file_qs, "QS (Training Set)")

# TEST : 100% de THE
X_test_text, y_test = load_and_label(file_the, "THE (Testing Set)")

# VECTORISATION ET ENTRAÎNEMENT

print("\n=== 2. APPRENTISSAGE SUR QS ===")

# On apprend le vocabulaire UNIQUEMENT sur QS
tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
X_train_vec = tfidf.fit_transform(X_train_text)

# On transforme THE avec le vocabulaire de QS (les mots inconnus de THE seront ignorés)
X_test_vec = tfidf.transform(X_test_text)

# Entraînement du modèle
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

print("-> Le modèle a appris à reconnaître les valeurs selon la vision de QS.")

# ÉVALUATION SUR THE
print("\n=== 3. TEST DE TRANSFÉRABILITÉ SUR THE ===")
print("L'IA (entraînée sur QS) essaie de deviner les catégories des unifs THE...")

y_pred = model.predict(X_test_vec)

# A. Score Global
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPRÉCISION GLOBALE : {accuracy*100:.2f}%")
print("(Est-ce que le modèle QS comprend le langage THE ?)")

# B. Rapport
print("\n--- RAPPORT PAR CATÉGORIE ---")
print(classification_report(y_test, y_pred))

# C. Matrice de Confusion
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred, labels=model.classes_), 
            annot=True, fmt='d', cmap='Purples', 
            xticklabels=model.classes_, yticklabels=model.classes_)

plt.title(f'Modèle Entraîné sur QS -> Testé sur THE\nAccuracy: {accuracy*100:.1f}%')
plt.ylabel('Vraie Catégorie (THE)')
plt.xlabel('Prédiction du Modèle (Vision QS)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ANALYSE DES ERREURS
# Si le score est bas, voyons où ça coince
if accuracy < 0.7:
    print("\n/!\\ Analyse : Le transfert est difficile.")
    print("Cela signifie que QS et THE utilisent des mots différents pour décrire les mêmes valeurs.")
else:
    print("\n Analyse : Le transfert est réussi.")
    print("Le vocabulaire des valeurs est standardisé entre les deux classements.")







# ==============================================================================
# 2 - Une méthode de clustering (non-supervisée)

'''