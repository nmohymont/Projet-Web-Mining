import json
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

# ==============================================================================
# 1. CONFIGURATION ET CHARGEMENT
# ==============================================================================

files_to_analyze = [
    'DATA/CLEAN/JSON/donnees_traitees_qs.json',
    'DATA/CLEAN/JSON/donnees_traitees_the.json'
]

# --- PARAMÈTRES DU GRAPHE ---
# On prend un peu plus de mots pour avoir un réseau intéressant
TOP_N_WORDS = 50  
# On ne garde que les liens forts pour éviter que tout soit connecté à tout
MIN_CO_OCCURRENCE = 200 

def load_data(files):
    all_tokens = []
    for file_path in files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for tokens in data.get('tokens', {}).values():
                    all_tokens.append(tokens)
    return all_tokens

print("=== 1. CONSTRUCTION DU GRAPHE ===")
docs = load_data(files_to_analyze)

# 1. Identifier les mots les plus fréquents (Nœuds)
flat_tokens = [t for doc in docs for t in doc]
most_common = dict(Counter(flat_tokens).most_common(TOP_N_WORDS))
top_words = set(most_common.keys())

# 2. Calculer les co-occurrences (Liens)
co_occurrences = Counter()
for doc in docs:
    # On ne garde que les mots du Top N présents dans ce document
    filtered = [t for t in doc if t in top_words]
    unique_tokens = sorted(list(set(filtered)))
    if len(unique_tokens) > 1:
        co_occurrences.update(combinations(unique_tokens, 2))

# 3. Créer le graphe NetworkX
G = nx.Graph()

# Ajout des nœuds
for word in top_words:
    G.add_node(word)

# Ajout des liens (seulement si poids > seuil)
for (w1, w2), weight in co_occurrences.items():
    if weight >= MIN_CO_OCCURRENCE:
        G.add_edge(w1, w2, weight=weight)

print(f"-> Graphe créé : {G.number_of_nodes()} nœuds, {G.number_of_edges()} liens.")

# ==============================================================================
# 2. CALCUL DES MESURES (LINK ANALYSIS)
# ==============================================================================
print("\n=== 2. CALCUL DES MÉTRIQUES (LINK ANALYSIS) ===")

# A. DEGREE CENTRALITY (Le Hub)
# Indique le nombre de connexions. Un score élevé = un mot "carrefour".
degree_dict = nx.degree_centrality(G)

# B. PAGERANK (L'Influence)
# Indique l'importance d'un mot en fonction de l'importance de ses voisins.
pagerank_dict = nx.pagerank(G, weight='weight')

# C. BETWEENNESS CENTRALITY (Le Pont)
# Indique si un mot sert de passage obligé entre deux clusters de mots.
betweenness_dict = nx.betweenness_centrality(G, weight='weight')

# D. SHORTEST PATH (La Distance)
# On calcule le chemin moyen du graphe (si connexe)
if nx.is_connected(G):
    avg_path_len = nx.average_shortest_path_length(G, weight=None) # Hop count
    diameter = nx.diameter(G)
    print(f"-> Chemin le plus court moyen (Average Path Length) : {avg_path_len:.2f} sauts")
    print(f"-> Diamètre du réseau (Distance max) : {diameter} sauts")
else:
    # Si le graphe n'est pas connecté, on prend la plus grande composante
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)
    avg_path_len = nx.average_shortest_path_length(subgraph)
    print(f"-> (Graphe non connexe) Chemin moyen sur la composante principale : {avg_path_len:.2f}")

# ==============================================================================
# 3. AFFICHAGE DES RÉSULTATS (TABLEAUX)
# ==============================================================================

def print_top_metric(metric_dict, name, top_n=10):
    sorted_metric = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\n>>> TOP {top_n} - {name.upper()} <<<")
    print("-" * 40)
    for i, (node, score) in enumerate(sorted_metric, 1):
        print(f"{i}. {node:<20} : {score:.4f}")

print_top_metric(degree_dict, "Degree Centrality (Les Connecteurs)")
print_top_metric(pagerank_dict, "PageRank (Les Stars)")
print_top_metric(betweenness_dict, "Betweenness Centrality (Les Ponts)")

# ==============================================================================
# 4. VISUALISATION
# ==============================================================================
print("\n=== 3. GÉNÉRATION DE LA CARTE D'INFLUENCE ===")

plt.figure(figsize=(15, 12))

# Layout : Spring layout écartelé pour lisibilité
pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

# TAILLE des nœuds basée sur PAGERANK
node_sizes = [pagerank_dict[node] * 10000 for node in G.nodes()]

# COULEUR des nœuds basée sur BETWEENNESS (Plus c'est foncé, plus c'est un pont)
node_colors = [betweenness_dict[node] for node in G.nodes()]

# Dessin
nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                               cmap=plt.cm.viridis, alpha=0.9, edgecolors='black')
nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

# Barre de couleur
plt.colorbar(nodes, label='Betweenness Centrality (Rôle de pont)')

plt.title("Link Analysis : Taille = PageRank (Importance), Couleur = Betweenness (Pont)", fontsize=16)
plt.axis('off')

# Légende textuelle pour le rapport
plt.figtext(0.5, 0.02, 
            f"Analyse sur les {TOP_N_WORDS} mots les plus fréquents.\n"
            f"Liens tracés si co-occurrence >= {MIN_CO_OCCURRENCE}.", 
            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})

plt.show()