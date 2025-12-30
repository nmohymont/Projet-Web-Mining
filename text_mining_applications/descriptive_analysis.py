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



# ==============================================================================
# 1 - Nuage de mot avant/après 2015 (VERSION 100% JSON)

# --- CONFIGURATION DES FICHIERS ---
# Chemins vers tes 4 fichiers JSON (Adaptez les chemins si besoin)
files_config = {
    'pre_2015': [
        'DATA/CLEAN/JSON/donnees_traitees_the_2012.json'  # Seul fichier avant 2015
    ],
    'post_2015': [
        'DATA/CLEAN/JSON/donnees_traitees_qs.json'      # 2025
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
                    print(f"   Avertissement : Aucune donnée 'tokens' trouvée dans {file_path}")
                    continue

                # 3. Agrégation (On ajoute les listes de mots de chaque université)
                count_unis = 0
                for tokens_list in docs_tokens.values():
                    aggregated_tokens.extend(tokens_list)
                    count_unis += 1
                
                print(f"   -> Chargé : {os.path.basename(file_path)} ({count_unis} universités)")
                
        except FileNotFoundError:
            print(f"    ATTENTION : Fichier introuvable -> {file_path}")
        except json.JSONDecodeError:
            print(f"    ERREUR JSON : Le fichier {file_path} est malformé ou corrompu.")
        except Exception as e:
            print(f"   Erreur inattendue sur {file_path} : {e}")
            
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

# ==============================================================================
# 5 - Analyse de concordance 

import pickle
import nltk
import pandas as pd
from nltk.text import Text

# CONFIGURATION
files_pkl = [
    'DATA/CLEAN/PKL/donnees_traitees_qs.pkl', 
    'DATA/CLEAN/PKL/donnees_traitees_the.pkl'
]

file_qs_raw = 'DATA/CLEAN/PARQUET/qs_university_corpus.parquet'
file_the_raw = 'DATA/CLEAN/PARQUET/the_university_corpus.parquet'

# Tes mots cibles
TARGET_WORDS = ["sustainable", "impact", "innovation"]

# PARTIE A : CONCORDANCE TECHNIQUE (Scan rapide)
# Utilise les données nettoyées (.pkl)
print("\n" + "="*50)
print("PARTIE A : CONCORDANCE VISUELLE (KWIC)")
print("="*50)

all_tokens_clean = []
for f_path in files_pkl:
    with open(f_path, 'rb') as f:
        data = pickle.load(f)
        for tokens in data[0].values():
            all_tokens_clean.extend(tokens)

text_object = Text(all_tokens_clean)

for word in TARGET_WORDS:
    print(f"\n--- Scan du mot : '{word}' ---")
    # Affiche le mot centré
    text_object.concordance(word, lines=5, width=90)


# PARTIE B : EXTRACTION DE SENS (Pour le Rapport)
# Utilise les textes bruts (.parquet) pour avoir de vraies phrases
print("\n" + "="*50)
print("PARTIE B : DÉFINITIONS ET PHRASES COMPLÈTES")
print("="*50)

# Chargement des vraies phrases
df_qs = pd.read_parquet(file_qs_raw)
df_the = pd.read_parquet(file_the_raw)
all_descriptions_raw = list(df_qs['description']) + list(df_the['description'])

for target in TARGET_WORDS:
    print(f"\n>>> CONTEXTE RÉEL POUR : '{target.upper()}'")
    
    found_sentences = []
    
    for desc in all_descriptions_raw:
        # On découpe en phrases grâce à NLTK
        sentences = nltk.sent_tokenize(str(desc))
        for sent in sentences:
            if target in sent.lower():
                found_sentences.append(sent.replace('\n', ' ').strip())
    
    # On affiche les 3 meilleures phrases (longues > 60 caractères pour éviter les titres)
    long_sentences = [s for s in found_sentences if len(s) > 60]
    
    if long_sentences:
        for i, s in enumerate(long_sentences[:3]):
            print(f"📖 Ex {i+1}: {s}\n")
    else:
        print("Aucune phrase longue trouvée.")

print("\nAnalyse terminée.")

# =============================================================================
# 6 - N-gram 

import pandas as pd
import nltk
from collections import Counter

# CONFIGURATION
# On utilise les fichiers PARQUET (Texte brut, avec "is", "the", "a"...)
files_raw = [
    'DATA/CLEAN/PARQUET/qs_university_corpus.parquet',
    'DATA/CLEAN/PARQUET/the_university_corpus.parquet'
]

# L'expression déclencheur (Le début de la phrase que tu cherches)
# Pour le marketing c'était "digital marketing is".
# Pour les unifs, essaye : "university is", "committed to", "located in", "leader in"
TRIGGER_PHRASE = "university is" 

# La longueur de la suite (14-grammes comme demandé)
N_GRAM_SIZE = 14

# CHARGEMENT ET PRÉPARATION
print("=== GÉNÉRATION DU DICTIONNAIRE AUTOMATIQUE ===")

full_text = []

# Chargement des textes bruts
for f in files_raw:
    try:
        df = pd.read_parquet(f)
        # On met tout en minuscule pour la recherche, mais on garde la structure
        full_text.extend(df['description'].astype(str).tolist())
    except FileNotFoundError:
        print(f"Fichier introuvable : {f}")

print(f"Analyse sur {len(full_text)} descriptions d'universités.")

# ALGORITHME D'EXTRACTION DE N-GRAMMES
print(f"\nRecherche des {N_GRAM_SIZE}-grammes suivant l'expression : '{TRIGGER_PHRASE}'...")

found_sequences = []
trigger_tokens = TRIGGER_PHRASE.lower().split()
len_trigger = len(trigger_tokens)

for desc in full_text:
    # Tokenisation simple (garde la ponctuation pour le sens, ou l'enlève selon préférence)
    # Ici on utilise une méthode rapide qui garde les mots
    tokens = nltk.word_tokenize(desc.lower())
    
    # On parcourt les mots pour trouver le déclencheur
    for i in range(len(tokens) - len_trigger - N_GRAM_SIZE):
        # Si on trouve la séquence déclencheur (ex: "university", "is")
        if tokens[i : i + len_trigger] == trigger_tokens:
            # On capture les N mots qui suivent
            sequence = tokens[i + len_trigger : i + len_trigger + N_GRAM_SIZE]
            # On rejoint en phrase
            found_sequences.append(" ".join(sequence))

# RÉSULTATS ET AFFICHAGE

# Comptage des plus fréquents
# Note : Sur des textes d'universités (très variés), il est possible que les 14-grammes
# soient uniques. Si c'est le cas, on affichera juste les premiers trouvés.
counts = Counter(found_sequences)
top_10 = counts.most_common(10)

print(f"\n--- TOP 10 SÉQUENCES APRÈS '{TRIGGER_PHRASE.upper()}' ---\n")

if not top_10:
    print("Aucune séquence trouvée. Essayez une expression plus courante (ex: 'located in').")
else:
    for i, (seq, count) in enumerate(top_10):
        print(f"{i+1}. [{count} x] ... {seq} ...")

# Si les fréquences sont toutes à 1 (ce qui arrive avec 14 mots), 
# c'est que les phrases sont trop uniques. Le code conseille alors de réduire N.
if top_10 and top_10[0][1] == 1:
    print("\n  Note : Les fréquences sont basses. Pour voir des motifs récurrents,")
    print("essayez de réduire N_GRAM_SIZE à 5 ou 6, ou changez le TRIGGER_PHRASE.")


# ==============================================================================
# 7 - Evolution temporelle des termes
# ==============================================================================

# ==============================================================================
# 7 - Évolution temporelle des termes ODD (Source THE - Top 200 uniquement)
# ==============================================================================

# 1. CONFIGURATION

# Liste temporelle des fichiers : UNIQUEMENT THE
files_timeline = [
    {'year': '2012', 'source': 'THE', 'path': 'DATA/CLEAN/JSON/donnees_traitees_the_2012.json'},
    {'year': '2021', 'source': 'THE', 'path': 'DATA/CLEAN/JSON/donnees_traitees_the_2021.json'},
    {'year': '2025', 'source': 'THE', 'path': 'DATA/CLEAN/JSON/donnees_traitees_the.json'}
]

# THEMES À ANALYSER : Termes liés aux ODD (Objectifs de Développement Durable)
# Suggestions : sustainable, environment, climate, social, equality, health, poverty
KEYWORDS = ["sustainable", "environment", "climate", "social", "impact", "development", "equality"]

# Limite : Seulement les 200 premières universités
TOP_N_LIMIT = 200

# CALCUL DES FRÉQUENCES RELATIVES
print(f"=== CALCUL DE L'ÉVOLUTION TEMPORELLE (THE - Top {TOP_N_LIMIT}) ===")

results = []

for item in files_timeline:
    label = item['year'] 
    path = item['path']
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # 1. Chargement JSON
            data = json.load(f)
            
            # 2. Récupération des tokens
            docs_tokens_all = data.get('tokens', {})
            
            # 3. FILTRAGE : On ne garde que les 200 premières universités
            # On transforme le dictionnaire en liste de paires, on coupe à 200, et on refait un dictionnaire
            # (Cela suppose que le JSON a été sauvegardé dans l'ordre du classement, ce qui est généralement le cas)
            docs_tokens_limit = dict(list(docs_tokens_all.items())[:TOP_N_LIMIT])
            
            nb_unis = len(docs_tokens_limit)
            
            # 4. On rassemble tous les mots de ces 200 universités
            all_tokens = []
            for tokens_list in docs_tokens_limit.values():
                all_tokens.extend(tokens_list)
            
            total_words_count = len(all_tokens)
            
            # 5. On compte les occurrences
            counts = Counter(all_tokens)
            
            # 6. Pour chaque mot-clé, on calcule sa fréquence normalisée
            row = {'Year_Label': label, 'Year_Int': int(item['year'])}
            
            for word in KEYWORDS:
                if total_words_count > 0:
                    # Fréquence pour 10 000 mots
                    freq = (counts.get(word, 0) / total_words_count) * 10000
                else:
                    freq = 0
                row[word] = freq
            
            results.append(row)
            print(f"-> Traité : {label} ({nb_unis} universités analysées, {total_words_count} mots)")

    except FileNotFoundError:
        print(f"/!\\ ERREUR : Fichier introuvable {path}")
    except json.JSONDecodeError:
        print(f"/!\\ ERREUR : JSON corrompu {path}")
    except Exception as e:
        print(f"/!\\ ERREUR sur {path} : {e}")

# Création du DataFrame
df_trends = pd.DataFrame(results)

# VISUALISATION (LINE CHART)
print("\n=== GÉNÉRATION DU GRAPHIQUE ===")

if df_trends.empty:
    print("Aucune donnée n'a pu être chargée.")
else:
    # Configuration du style
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # On trace une ligne pour chaque mot
    for word in KEYWORDS:
        # linewidth=3 pour bien voir les lignes
        sns.lineplot(data=df_trends, x='Year_Label', y=word, marker='o', linewidth=3, label=word)

    plt.title(f"Évolution des termes ODD (Top {TOP_N_LIMIT} Universités THE)", fontsize=16, fontweight='bold')
    plt.ylabel("Fréquence (pour 10k mots)", fontsize=12)
    plt.xlabel("Année", fontsize=12)

    # Légende à l'extérieur
    plt.legend(title="Termes ODD", title_fontsize='12', fontsize='11', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

    # TABLEAU DE DONNÉES (POUR LE RAPPORT)
    print("\n--- TABLEAU DES VALEURS (Fréquence / 10k mots) ---")
    print(df_trends.set_index('Year_Label')[KEYWORDS].round(2))