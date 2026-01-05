import json
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
import os
import matplotlib.pyplot as plt # Import nécessaire pour la visualisation

# ==============================================================================
# 1. CONFIGURATION ET CHARGEMENT
# ==============================================================================

# À ADAPTER : Chemin vers votre fichier JSON nettoyé
FILE_PATH = 'DATA/CLEAN/JSON/university_processed_features_qs.json'
# FILE_PATH = 'university_processed_features_qs.json' # Si dans le même dossier

# Limite de tokens par université pour alléger le calcul
TOP_N_TOKENS = 100 

# Seuil de probabilité pour créer un lien (0.25 = Le mot B apparait dans 25% des cas où A est là)
THRESHOLD_DIR = 0.8

def load_data(filepath, top_n):
    if not os.path.exists(filepath):
        print(f"ERREUR : Le fichier {filepath} est introuvable.")
        return []

    print(f"Chargement de {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    raw_tokens = data.get('tokens', {})
    
    # On garde seulement les N premiers tokens pour chaque université
    filtered_list = [tokens[:top_n] for tokens in raw_tokens.values() if tokens]
    
    print(f"-> {len(filtered_list)} universités chargées.")
    return filtered_list

# ==============================================================================
# 2. CALCUL DE LA MATRICE ASYMÉTRIQUE (Probabilité Conditionnelle)
# ==============================================================================

def build_conditional_matrix(tokens_list):
    """
    Construit une matrice dirigée où M[i, j] = P(Mot_j | Mot_i)
    """
    print("\nConstruction de la matrice binaire (Document-Terme)...")
    mlb = MultiLabelBinarizer(sparse_output=True)
    
    A_sparse = mlb.fit_transform(tokens_list)
    vocab = mlb.classes_
    print(f"-> Vocabulaire détecté : {len(vocab)} mots.")

    X = A_sparse.T
    
    print("Calcul des co-occurrences...")
    cooccurrence_matrix = (X * X.T).toarray()
    
    word_freqs = cooccurrence_matrix.diagonal()
    
    print("Calcul des probabilités conditionnelles...")
    # P(B | A) = Cooccurrence(A, B) / Fréquence(A)
    with np.errstate(divide='ignore', invalid='ignore'):
        conditional_prob = cooccurrence_matrix / word_freqs[:, None]
        conditional_prob = np.nan_to_num(conditional_prob)
    
    np.fill_diagonal(conditional_prob, 0)
    
    return conditional_prob, vocab

# ==============================================================================
# 3. CONSTRUCTION DU GRAPHE DIRIGÉ
# ==============================================================================

def create_directed_graph(matrix, vocabulary, threshold):
    G = nx.DiGraph()
    rows, cols = np.where(matrix > threshold)
    
    edge_list = []
    print(f"\nCréation des arêtes (Seuil > {threshold})...")
    
    for i, j in zip(rows, cols):
        weight = matrix[i, j]
        edge_list.append((vocabulary[i], vocabulary[j], weight))
    
    G.add_weighted_edges_from(edge_list)
    print(f"-> Graphe généré : {G.number_of_nodes()} nœuds, {G.number_of_edges()} liens dirigés.")
    return G

# ==============================================================================
# 4. EXÉCUTION PRINCIPALE
# ==============================================================================

tokens_data = load_data(FILE_PATH, TOP_N_TOKENS)

if tokens_data:
    # --- Création du Graphe ---
    prob_matrix, vocab = build_conditional_matrix(tokens_data)
    G_dir = create_directed_graph(prob_matrix, vocab, THRESHOLD_DIR)
    
    # --- Calculs des Métriques (Nécessaires pour l'export et la visu) ---
    print("\nCalcul des métriques...")
    
    # 1. PageRank (Importance structurelle)
    try:
        pagerank = nx.pagerank(G_dir, alpha=0.85, weight='weight')
    except:
        pagerank = {n: 0 for n in G_dir.nodes()} # Fallback si erreur

    # 2. In-Degree (Popularité / Cible) - C'EST CE QUI MANQUAIT
    in_degree = dict(G_dir.in_degree(weight='weight'))

    # 3. HITS (Hubs & Authorities)
    try:
        hubs, authorities = nx.hits(G_dir, max_iter=100, normalized=True) #HITS
    except:
        hubs, authorities = {}, {}

    # --- Affichage Console des Tops ---
    print("\n--- TOP 5 PAGERANK (Concepts Structurants) ---")
    sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
    for node, score in sorted_pr:
        print(f"   - {node} ({score:.4f})")

    # --- Export Gephi (.gexf) ---
    output_file = "graphe_semantique_dirige.gexf"
    print(f"\nExportation vers {output_file}...")
    
    # On ajoute les attributs aux nœuds pour qu'ils soient dans Gephi
    nx.set_node_attributes(G_dir, pagerank, 'pagerank')
    nx.set_node_attributes(G_dir, in_degree, 'in_degree')
    nx.set_node_attributes(G_dir, hubs, 'hub_score')
    nx.set_node_attributes(G_dir, authorities, 'authority_score')
    
    nx.write_gexf(G_dir, output_file)
    print("Export terminé.")

    