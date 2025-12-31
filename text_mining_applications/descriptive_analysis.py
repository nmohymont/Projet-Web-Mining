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

import os

import nltk
from nltk.text import Text



# ==============================================================================
# 1 - Nuage de mot avant/après 2015 (VERSION 100% JSON)

# --- CONFIGURATION DES FICHIERS ---
files_config = {
    'pre_2015': [
        'DATA/CLEAN/JSON/donnees_traitees_the_2012.json'  # Avant 2015
    ],
    'post_2015': [
        'DATA/CLEAN/JSON/donnees_traitees_qs.json',       # ATTENTION: J'ai ajouté la virgule manquante ici
        'DATA/CLEAN/JSON/donnees_traitees_the.json',
        'DATA/CLEAN/JSON/donnees_traitees_the_2021.json'
    ]
}

# --- FONCTION DE CHARGEMENT ---
def load_and_aggregate_tokens(file_list):
    """Charge plusieurs fichiers JSON et combine tous les tokens."""
    aggregated_tokens = []
    
    for file_path in file_list:
        try:
            if not os.path.exists(file_path):
                print(f"   /!\\ Fichier introuvable (ignoré) : {file_path}")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                docs_tokens = data.get("tokens", {})
                
                if not docs_tokens:
                    continue

                count_unis = 0
                for tokens_list in docs_tokens.values():
                    # On peut filtrer ici si on veut (ex: prendre que les 50 premiers mots)
                    aggregated_tokens.extend(tokens_list)
                    count_unis += 1
                
                print(f"   -> Chargé : {os.path.basename(file_path)} ({count_unis} universités)")
                
        except Exception as e:
            print(f"   Erreur sur {file_path} : {e}")
            
    return aggregated_tokens

# --- EXÉCUTION ---
print("=== CHARGEMENT ET AGRÉGATION ===")

print("\n1. Corpus 'HÉRITAGE' (Avant 2015)...")
tokens_pre_2015 = load_and_aggregate_tokens(files_config['pre_2015'])

print("\n2. Corpus 'RESPONSABILITÉ' (Après 2015)...")
tokens_post_2015 = load_and_aggregate_tokens(files_config['post_2015'])

print(f"\nTotal mots Avant : {len(tokens_pre_2015)}")
print(f"Total mots Après : {len(tokens_post_2015)}")

# --- VISUALISATION (NUAGES DE MOTS) ---
def plot_compare_wordclouds(tokens1, tokens2, title1, title2):
    if not tokens1 or not tokens2:
        print("Données insuffisantes pour les nuages.")
        return

    text1 = " ".join(tokens1)
    text2 = " ".join(tokens2)

    # --- CHOIX DES COULEURS ---
    # 'magma' : Noir/Rouge/Orange (Prestige, Sérieux, Histoire)
    # 'viridis' : Violet/Vert/Jaune (Moderne, Nature, Lisible)
    
    wc1 = WordCloud(width=800, height=500, background_color='white', 
                   collocations=False, max_words=60, 
                   colormap='inferno').generate(text1) # Essayez 'magma' ou 'inferno'
                   
    wc2 = WordCloud(width=800, height=500, background_color='white', 
                   collocations=False, max_words=60, 
                   colormap='viridis').generate(text2) # Essayez 'viridis' ou 'ocean'

    # Affichage
    fig, axes = plt.subplots(1, 2, figsize=(22, 12))

    # Nuage 1
    axes[0].imshow(wc1, interpolation='bilinear')
    axes[0].set_title(f"{title1}\n(L'Ancien Monde)", fontsize=18, fontweight='bold', color="#af0303") # Rouge foncé
    axes[0].axis('off')

    # Nuage 2
    axes[1].imshow(wc2, interpolation='bilinear')
    axes[1].set_title(f"{title2}\n(Le Nouveau Monde)", fontsize=18, fontweight='bold', color='#004d40') # Vert foncé
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

print("\n=== GÉNÉRATION DES NUAGES DE MOTS ===")
plot_compare_wordclouds(tokens_pre_2015, tokens_post_2015, "Avant 2015", "Après 2015")

# ==============================================================================
# 2 - Nuage de mot en fonction des continents 


# --- CONFIGURATION ---
json_files = [
    'DATA/CLEAN/JSON/donnees_traitees_qs.json', 
    'DATA/CLEAN/JSON/donnees_traitees_the.json',
    'DATA/CLEAN/JSON/donnees_traitees_the_2012.json', 
    'DATA/CLEAN/JSON/donnees_traitees_the_2021.json'
]

# Dictionnaire global
tokens_by_continent = {}

print("=== CHARGEMENT ET AGRÉGATION (MODE JSON PUR) ===")

for json_path in json_files:
    try:
        print(f"-> Lecture de {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 1. On récupère les tokens
            docs_tokens = data.get('tokens', {})
            
            # 2. On récupère le mapping des régions
            mapping_regions = data.get('regions', {}) 
            
            if not mapping_regions:
                print(f"   /!\\ ATTENTION : Pas de clé 'regions' trouvée dans {json_path}")
                continue

            # 3. Distribution des mots
            count_matched = 0
            for uni_name, tokens in docs_tokens.items():
                region = mapping_regions.get(uni_name, "Inconnu")
                
                if region is None or str(region).lower() == 'nan':
                    region = "Inconnu"
                
                if region != "Inconnu":
                    if region not in tokens_by_continent:
                        tokens_by_continent[region] = []
                    tokens_by_continent[region].extend(tokens)
                    count_matched += 1
            
            print(f"   {count_matched} universités localisées.")

    except Exception as e:
        print(f"   /!\\ Erreur sur {json_path} : {e}")

# PLOT 1 : VUE D'ENSEMBLE (TOUTES LES RÉGIONS)
print("\n=== PLOT 1 : VUE D'ENSEMBLE (TOUTES LES RÉGIONS) ===")

regions_to_plot = [r for r, t in tokens_by_continent.items() if len(t) > 50]
regions_to_plot.sort()

if not regions_to_plot:
    print("Aucune donnée.")
else:
    nb_regions = len(regions_to_plot)
    cols = 3
    rows = math.ceil(nb_regions / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    
    # Gestion axes
    if nb_regions > 1: axes_flat = axes.flatten()
    else: axes_flat = [axes]

    for i, region in enumerate(regions_to_plot):
        ax = axes_flat[i]
        tokens = tokens_by_continent[region]
        text = " ".join(tokens)
        
        wc = WordCloud(width=800, height=500, background_color='white', collocations=False, 
                       max_words=15, min_font_size=15, colormap='Dark2').generate(text)
        
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f"{region.upper()}\n({len(tokens)} mots)", fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
        
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_edgecolor('#cccccc')

    # Masquer les vides
    for j in range(nb_regions, len(axes_flat)):
        axes_flat[j].axis('off')
        axes_flat[j].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.show()

print("\n=== PLOT 2 : ZOOM STRATÉGIQUE (NA, EUROPE, ASIA) ===")

target_regions = ["North America", "Europe", "Asia"]
# Nombre de mots à afficher (dans le nuage ET dans la console)
TOP_N_ZOOM = 30 

regions_subset = [r for r in target_regions if r in tokens_by_continent]

if not regions_subset:
    print("Aucune des régions cibles (NA, EU, ASIA) n'a été trouvée dans les données.")
else:
    # On crée une figure avec 3 colonnes fixes
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 7)) 
    
    if len(regions_subset) == 1: axes2 = [axes2]
    
    for i, region in enumerate(regions_subset):
        ax = axes2[i]
        tokens = tokens_by_continent[region]
        
        # --- AJOUT : CALCUL ET AFFICHAGE CONSOLE ---
        # 1. On compte les mots
        counts = Counter(tokens)
        # 2. On prend les X plus fréquents
        top_words = counts.most_common(TOP_N_ZOOM)
        
        # 3. On affiche dans la console
        print(f"\n>>> LISTE DES MOTS : {region.upper()} (Top {TOP_N_ZOOM}) <<<")
        print("-" * 40)
        # On affiche sous forme de tableau propre : Rang | Mot | Fréquence
        for rank, (word, freq) in enumerate(top_words, 1):
            print(f"{rank:02d}. {word:<20} ({freq} occurrences)")
        print("-" * 40)
        # -------------------------------------------

        text = " ".join(tokens)
        
        # On génère le nuage
        wc = WordCloud(
            width=800, 
            height=500, 
            background_color='white', 
            collocations=False, 
            max_words=TOP_N_ZOOM, # On utilise la même limite que l'affichage console
            min_font_size=12, 
            colormap='tab10'
        ).generate(text)
        
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f"--- {region.upper()} ---", fontsize=18, fontweight='bold', color='darkblue', pad=15)
        ax.axis('off')
        
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(2); spine.set_edgecolor('#333333')

    # Masquer les axes vides
    for j in range(len(regions_subset), 3):
        axes2[j].axis('off')
        axes2[j].set_visible(False)

    plt.tight_layout(pad=3.0)
    plt.show()

# ==============================================================================
# 3 - Comparaison mots fichier the et fichier qs

'''

# Chemins des fichiers
# Assure-toi que le CSV est bien le fichier de mapping (correspondance entre noms QS et THE)
file_matches = 'DATA/CLEAN/CSV/university_mapping_qs_the.csv'

# Nouveaux chemins vers les JSON
path_json_qs = 'DATA/CLEAN/JSON/donnees_traitees_qs.json'
path_json_the = 'DATA/CLEAN/JSON/donnees_traitees_the.json'

# Seuil de matching (Score de similarité du nom)
SCORE_THRESHOLD = 0.857

# Noms des colonnes du CSV
COL_QS_NAME = 'QS_Name'   
COL_THE_NAME = 'THE_Name'
COL_SCORE = 'Score'

# ==============================================================================
# 1. CHARGEMENT DES DONNÉES
# ==============================================================================
print("=== 1. CHARGEMENT ===")

# A. Chargement et filtrage du CSV (Le "Juge")
try:
    df_matches = pd.read_csv(file_matches)
    # On ne garde que les lignes avec un bon score de correspondance
    df_filtered = df_matches[df_matches[COL_SCORE] > SCORE_THRESHOLD]
    print(f"-> CSV chargé. Paires valides (> {SCORE_THRESHOLD}) : {len(df_filtered)}")
except FileNotFoundError:
    print(f"Erreur : Impossible de trouver le fichier CSV {file_matches}")
    exit()

# B. Fonction de chargement JSON
def load_json_tokens(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # IMPORTANT : Dans ton JSON, les mots sont sous la clé "tokens"
            # Structure : { "info": ..., "tokens": { "Univ": ["mot1"] }, ... }
            return data.get('tokens', {}) 
    except FileNotFoundError:
        print(f"Erreur : Impossible de trouver {path}")
        return {}
    except json.JSONDecodeError:
        print(f"Erreur : Le fichier {path} n'est pas un JSON valide.")
        return {}

print("-> Chargement du JSON QS...")
docs_qs = load_json_tokens(path_json_qs)

print("-> Chargement du JSON THE...")
docs_the = load_json_tokens(path_json_the)


# ==============================================================================
# 2. CROISEMENT ET FILTRAGE
# ==============================================================================
print("\n=== 2. CROISEMENT DES DONNÉES ===")

tokens_qs_final = []
tokens_the_final = []
count_match = 0
missing_qs = 0
missing_the = 0

# On parcourt le CSV filtré
for index, row in df_filtered.iterrows():
    name_qs = str(row[COL_QS_NAME]).strip()
    name_the = str(row[COL_THE_NAME]).strip()
    
    # On vérifie si l'université existe bien dans nos fichiers JSON
    # (On utilise .get() pour éviter les erreurs si la clé n'existe pas)
    words_qs = docs_qs.get(name_qs)
    words_the = docs_the.get(name_the)

    if words_qs and words_the:
        # Si on a trouvé l'université dans les DEUX fichiers JSON
        tokens_qs_final.extend(words_qs)
        tokens_the_final.extend(words_the)
        count_match += 1
    else:
        # Juste pour le debug, voir pourquoi ça ne matche pas
        if not words_qs: missing_qs += 1
        if not words_the: missing_the += 1

print(f"-> Analyse basée sur {count_match} universités communes (présentes dans CSV + JSONs).")
if missing_qs > 0 or missing_the > 0:
    print(f"-> Attention : {missing_qs} universités QS et {missing_the} universités THE du CSV n'ont pas été trouvées dans les JSON (problème de nom exact ?).")

print(f"-> Total mots QS  : {len(tokens_qs_final)}")
print(f"-> Total mots THE : {len(tokens_the_final)}")


# ==============================================================================
# 3. VISUALISATION
# ==============================================================================
print("\n=== 3. GÉNÉRATION DES NUAGES DE MOTS ===")

def plot_clouds(tokens1, tokens2):
    # Transformation liste -> texte
    text1 = " ".join(tokens1)
    text2 = " ".join(tokens2)
    
    if not text1 or not text2:
        print("Erreur : Pas assez de mots pour générer les nuages.")
        return

    # Création des WordClouds
    # On limite à 50 mots pour la lisibilité
    wc_qs = WordCloud(width=800, height=400, background_color='white', 
                      colormap='Blues', collocations=False, max_words=50).generate(text1)
                      
    wc_the = WordCloud(width=800, height=400, background_color='white', 
                       colormap='Reds', collocations=False, max_words=50).generate(text2)

    # Affichage
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    axes[0].imshow(wc_qs, interpolation='bilinear')
    axes[0].set_title(f"Vocabulaire QS 2025\n(Sur les {count_match} universités communes)", fontsize=16, color='darkblue', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(wc_the, interpolation='bilinear')
    axes[1].set_title(f"Vocabulaire THE 2025\n(Sur les {count_match} universités communes)", fontsize=16, color='darkred', fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Lancement
if count_match > 0:
    plot_clouds(tokens_qs_final, tokens_the_final)
else:
    print("Aucune correspondance trouvée. Vérifiez que les noms dans le CSV sont EXACTEMENT les mêmes que dans les clés du JSON.")

'''
# =============================================================================
# 4 - Graphe de co-occurrence combiné (QS + THE)

# CONFIGURATION

# Liste des fichiers JSON à combiner (Uniquement QS et THE comme demandé)
files_to_combine = [
    'DATA/CLEAN/JSON/donnees_traitees_qs.json',
    'DATA/CLEAN/JSON/donnees_traitees_the.json',
]

# --- PARAMÈTRES DU GRAPHE ---
# Nombre de mots les plus fréquents à afficher.
TOP_N_WORDS = 20

# Seuil minimum de cooccurrence
# Un lien est tracé seulement si les deux mots apparaissent ensemble dans X documents
MIN_EDGE_WEIGHT = 10 

# CHARGEMENT ET AGREGATION

print("=== 1. CHARGEMENT ET FUSION DES CORPUS (QS + THE) ===")

all_docs_list = [] # Liste qui contiendra les listes de mots de TOUTES les universités

for file_path in files_to_combine:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 1. Chargement du JSON
            data = json.load(f)
            
            # 2. Récupération des tokens
            docs_tokens = data.get('tokens', {})
            
            # 3. Ajout à la liste globale
            count_local = 0
            for tokens in docs_tokens.values():
                if tokens: # On évite les listes vides
                    all_docs_list.append(tokens)
                    count_local += 1
                
            print(f"-> Chargé avec succès : {os.path.basename(file_path)} ({count_local} universités)")
        
    except FileNotFoundError:
        print(f" /!\\ Erreur : Fichier introuvable -> {file_path}")
    except json.JSONDecodeError:
        print(f" /!\\ Erreur : JSON corrompu -> {file_path}")
    except Exception as e:
        print(f" /!\\ Erreur sur {file_path} : {e}")

print(f"\n-> TOTAL DOCUMENTS ANALYSÉS : {len(all_docs_list)}")

# CALCULS STATISTIQUES
print("\n=== 2. ANALYSE DES FRÉQUENCES ET COOCCURRENCES ===")

if len(all_docs_list) == 0:
    print("Erreur : Aucun document chargé. Vérifiez vos chemins de fichiers.")
    exit()

# A. Sélection des Top Mots sur l'ensemble combiné
all_tokens_flat = [token for doc in all_docs_list for token in doc]
word_counts = Counter(all_tokens_flat)

# On récupère les N mots les plus fréquents
top_words_dict = dict(word_counts.most_common(TOP_N_WORDS))
top_words_set = set(top_words_dict.keys())

print(f"-> Top {TOP_N_WORDS} mots sélectionnés (ex: {list(top_words_dict.keys())[:5]}...)")

# B. Calcul des Cooccurrences
co_occurrence_counts = Counter()

for tokens in all_docs_list:
    # 1. On ne garde que les mots du Top N présents dans ce document
    filtered_tokens = [t for t in tokens if t in top_words_set]
    
    # 2. Mots uniques par document
    unique_tokens = sorted(list(set(filtered_tokens)))
    
    # 3. Paires
    if len(unique_tokens) > 1:
        pairs = list(combinations(unique_tokens, 2))
        co_occurrence_counts.update(pairs)

print(f"-> Liens calculés. Total de paires uniques trouvées : {len(co_occurrence_counts)}")

# CONSTRUCTION DU GRAPHE (NetworkX)
print("\n=== 3. GÉNÉRATION DU GRAPHE COMBINÉ ===")

G = nx.Graph()

# A. Ajout des Nœuds
for word, count in top_words_dict.items():
    G.add_node(word, size=count)

# B. Ajout des Liens
edges_added = 0
for pair, weight in co_occurrence_counts.items():
    if weight >= MIN_EDGE_WEIGHT:
        G.add_edge(pair[0], pair[1], weight=weight)
        edges_added += 1

print(f"-> Graphe final : {G.number_of_nodes()} nœuds, {edges_added} liens.")

# VISUALISATION

plt.figure(figsize=(18, 14))

# 1. Disposition
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# 2. Tailles
if G.number_of_nodes() > 0:
    base_size = [G.nodes[n]['size'] for n in G.nodes]
    max_size = max(base_size) if base_size else 1
    node_sizes = [(s / max_size) * 3000 for s in base_size] 
else:
    node_sizes = []

if G.number_of_edges() > 0:
    weights = [G.edges[u, v]['weight'] for u, v in G.edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [(w / max_weight) * 4 for w in weights] 
else:
    edge_widths = []

# 3. Dessin
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#69b3a2', alpha=0.9, edgecolors='white')

# Labels (TAILLE AUGMENTÉE ICI)
nx.draw_networkx_labels(G, pos, font_size=14, font_family='sans-serif', font_weight='bold')

nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, edge_color='gray')

plt.title(f"Réseau de Cooccurrence (QS & THE) - Top {TOP_N_WORDS} mots", fontsize=20)
plt.axis('off')

plt.figtext(0.5, 0.02, 
            f"Basé sur {len(all_docs_list)} descriptions.\n"
            f"Lien affiché si cooccurrence >= {MIN_EDGE_WEIGHT}.", 
            ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

plt.show()

# -------------------------------------------------------------------------------
# 5 - Analyse de fréquence temporelle 

# --- CONFIGURATION ---
files_timeline = [
    {'year': '2012', 'source': 'THE', 'path': 'DATA/CLEAN/JSON/donnees_traitees_the_2012.json'},
    {'year': '2021', 'source': 'THE', 'path': 'DATA/CLEAN/JSON/donnees_traitees_the_2021.json'},
    {'year': '2025', 'source': 'THE', 'path': 'DATA/CLEAN/JSON/donnees_traitees_the.json'}
]

# ON GARDE L'ÉQUILIBRE DES UNIVERSITÉS
TOP_N_UNIV_LIMIT = 200 

# Mots à analyser
KEYWORDS_OLD = ["founded", "science", "teach"]
KEYWORDS_NEW = ["sustainable", "impact", "global", "collaboration", "innovation", "cultural", "people"]
KEYWORDS = KEYWORDS_OLD + KEYWORDS_NEW

# --- NOUVELLE PALETTE DE COULEURS ---
COLOR_MAP = {
    # VIEUX MONDE (Tons Chauds / Terre / Passé)
    "founded": "#d62728",      # Rouge brique
    "science": "#ff7f0e",      # Orange 
    "teach": "#8c564b",        # Marron terre

    # NOUVEAU MONDE (Tons Froids / Vifs / Futur)
    "sustainable": "#2ca02c",  # Vert (Écologie)
    "impact": "#1f77b4",       # Bleu standard (Action)
    "global": "#17becf",       # Cyan (International)
    "collaboration": "#9467bd",# Violet (Lien)
    "innovation": "#e377c2",   # Rose fuchsia (Modernité)
    "cultural": "#bcbd22",     # Jaune olive (Diversité)
    "people": "#000080"        # Bleu marine (L'Humain)
}

results = []
print("=== ANALYSE VOLUMÉTRIQUE (FULL TEXT) ===")

for item in files_timeline:
    try:
        with open(item['path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 1. On sélectionne l'Élite (Top 200)
            docs_tokens = dict(list(data.get('tokens', {}).items())[:TOP_N_UNIV_LIMIT])
            
            # 2. On prend TOUS les mots
            all_tokens = []
            for tokens in docs_tokens.values():
                all_tokens.extend(tokens)
            
            # 3. Normalisation (Base 10 000 mots)
            total_words = len(all_tokens)
            counts = Counter(all_tokens)
            
            row = {'Year_Label': item['year']}
            for word in KEYWORDS:
                if total_words > 0:
                    freq = (counts.get(word, 0) / total_words) * 10000 
                else:
                    freq = 0
                row[word] = freq
            results.append(row)
            
    except Exception as e:
        print(f"Erreur : {e}")

# --- VISUALISATION ---
df = pd.DataFrame(results)

if not df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    sns.set_style("whitegrid")

    # Graphique 1 : Ancrage (Vieux monde)
    for word in KEYWORDS_OLD:
        # On utilise .get() avec une couleur par défaut noire au cas où
        color = COLOR_MAP.get(word, '#000000')
        sns.lineplot(data=df, x='Year_Label', y=word, color=color, 
                     marker='o', linestyle='--', linewidth=3, ax=axes[0], label=word.upper())
    
    axes[0].set_title("L'Ancrage Historique & Académique (En Déclin)", fontsize=14, fontweight='bold', color='#8c564b')
    axes[0].set_ylabel("Occurrences pour 10 000 mots")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Graphique 2 : Impact (Nouveau monde)
    for word in KEYWORDS_NEW:
        color = COLOR_MAP.get(word, '#000000')
        sns.lineplot(data=df, x='Year_Label', y=word, color=color, 
                     marker='s', linestyle='-', linewidth=3, ax=axes[1], label=word.upper())
    
    axes[1].set_title("L'Impact Sociétal & Humain (En Hausse)", fontsize=14, fontweight='bold', color='#1f77b4')
    axes[1].set_ylabel("Occurrences pour 10 000 mots")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Le Grand Basculement Sémantique (2012-2025)", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()
    
    # Affichage des données brutes
    print("\n--- DONNÉES CALCULÉES (Fréq. / 10k mots) ---")
    print(df.set_index('Year_Label')[KEYWORDS].round(1))
else:
    print("Pas de données.")