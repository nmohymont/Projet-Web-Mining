











'''
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
    'DATA/CLEAN/JSON/university_processed_features_qs.json',
    'DATA/CLEAN/JSON/university_processed_features_the.json'
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

'''
import json
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer 
#instead of for loop based on the token stored in the JSON file per university usde MultiLabelBinazrier 
from sklearn.metrics import pairwise_distances
import sys
import os

# Ajoute le dossier courant au path pour que Python trouve le dossier 'imported_function'
sys.path.append(os.getcwd())

from imported_function.GraphMatrices import degree_matrix
from imported_function.CentralityMeasures import closeness_centrality, eccentricity_centrality
from imported_function.SimilarityMatrices import common_neighbors_matrix, katz_matrix


file_path = 'university_processed_features_qs.json'
top_n = 100

with open(file_path, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

# extract raw token from json file
raw_tokens_dict = full_data.get('tokens', {})

# --- C'est ici que tu remplaces/filtres directement ---
tokens_list = []
university_names = []

for name, tokens in raw_tokens_dict.items():
    university_names.append(name)
    # On coupe directement la liste ici
    tokens_list.append(tokens[:top_n]) 

print(f"Nombre d'universités extraites : {len(university_names)}")
print(f"Exemple pour {university_names[0]} : {tokens_list[0][:5]}")

def create_undirected_graph_jaccard(tokens_list):
    """
    Computes the Jaccard Similarity matrix for tokens based on their 
    co-occurrence across universities.
    
    Formula: J(i, j) = |Intersection(i, j)| / (|Degree(i)| + |Degree(j)| - |Intersection(i, j)|)
    
    Returns:
        jaccard_sim (np.ndarray): Dense matrix of similarities.
        vocab (np.ndarray): Array of token names corresponding to matrix indices.
    """
    # 1. Create the Document-Term Matrix (Binary)
    # This is your "A" matrix (Universities x Tokens)
    mlb = MultiLabelBinarizer(sparse_output=True)
    A_sparse = mlb.fit_transform(tokens_list)
    vocab = mlb.classes_

    print(f"Matrice A (Univ x Tokens) : {A_sparse.shape}")
    
    # 2. Transpose to get Token-University Matrix (X)
    X = A_sparse.T  # Tokens x Universities

    print(f"Matrice X (Tokens x Univ) : {X.shape}")
    
    # 3. Calculate Intersection (Numerator)
    # Matrix multiplication: Number of shared universities between token i and j
    intersection = (X * X.T).toarray()
    
    # 4. Calculate Degrees
    # Number of universities using each token
    token_degrees = np.array(X.sum(axis=1)).flatten()
    
    # 5. Calculate Union (Denominator) using broadcasting
    # deg_matrix[i, j] = degree(i) + degree(j)
    deg_matrix = np.add.outer(token_degrees, token_degrees)
    union_matrix = deg_matrix - intersection
    
    # 6. Final Jaccard Calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccard_sim = intersection / union_matrix
        # Handle cases where union is zero
        jaccard_sim = np.nan_to_num(jaccard_sim)
    
    # Remove self-similarity (diagonal) for network analysis
    np.fill_diagonal(jaccard_sim, 0)
    
    print(f"-> Jaccard Matrix computed: {jaccard_sim.shape} tokens.")
    return A_sparse, jaccard_sim, vocab

print("Matrice de similarité calculée !")

A_sparse, jaccard_sim, vocabulary = create_undirected_graph_jaccard(tokens_list)

# 4. Création du Graphe
G_tokens = nx.Graph()
THRESHOLD = 0.20

# On récupère les indices où Sim > Seuil
rows, cols = np.where(jaccard_sim > THRESHOLD)

edge_list = []
for i, j in zip(rows, cols):
    if i < j: # Triangle supérieur uniquement
        edge_list.append((vocabulary[i], vocabulary[j], jaccard_sim[i, j]))

G_tokens.add_weighted_edges_from(edge_list)

print(f"Graphe créé : {G_tokens.number_of_nodes()} noeuds, {G_tokens.number_of_edges()} arêtes.")

# 5. Export
#nx.write_gexf(G_tokens, "graphe_tokens_jaccard.gexf")

# Convertir le graphe en matrice d'adjacence numpy
# weight=None car le degré se calcule souvent sur la présence de lien (binaire), 
# mais tu peux mettre weight='weight' pour le degré pondéré (strength).
adj_matrix = nx.to_numpy_array(G_tokens, weight=None)
nodes = list(G_tokens.nodes())


D_out = degree_matrix(adj_matrix, direction="out")

degrees_list = np.diag(D_out)
print(f"Degré moyen : {np.mean(degrees_list)}")

top10_indices = np.argsort(degrees_list)[::-1][:10]

print("Top 10 des degrés sortants :")
for idx in top10_indices:
    print(f"noeud {nodes[idx]} -> degré = {degrees_list[idx]}")

closeness_vals = closeness_centrality(adj_matrix)

eccentricity_vals = eccentricity_centrality(adj_matrix)

cn_matrix = common_neighbors_matrix(adj_matrix)
np.fill_diagonal(cn_matrix,0)
ind_x, ind_y = np.unravel_index(np.argsort(cn_matrix, axis=None)[-10:], cn_matrix.shape)

# 4. Synthèse dans un DataFrame
df_analysis = pd.DataFrame({
    'Token': nodes,
    'Closeness': closeness_vals,
    'Eccentricity': eccentricity_vals,
    # 'Katz': katz_vals
})


# 5. Interprétation pour le rapport
print("--- LE NOYAU (Radius) ---")
min_ecc = df_analysis['Eccentricity'].min()
print(f"Radius du graphe : {min_ecc}")
print("Tokens au centre :", df_analysis[df_analysis['Eccentricity'] == min_ecc]['Token'].tolist())

print("\n--- LES PONTS (Top Closeness) ---")
print(df_analysis.sort_values('Closeness', ascending=False).head(5))

print("\n--- SYNONYMES CONTEXTUELS (Top Common Neighbors) ---")
for i, j in zip(ind_x, ind_y):
    print(f"{nodes[i]} <-> {nodes[j]} (Voisins communs: {cn_matrix[i,j]})")


def freeman_betweenness(G): # used to see what is used to caculate the betweenness
    """
    Implémentation littérale de la formule de Freeman.
    Note: Lent pour les grands graphes (O(N^3)). 
    Pour la prod, utiliser nx.betweenness_centrality (Brandes).
    """
    nodes = list(G.nodes())
    n = len(nodes)
    betweenness = {node: 0.0 for node in nodes}
    
    # Pour chaque paire unique (s, t)
    for i in range(n):
        for j in range(i + 1, n):
            s = nodes[i]
            t = nodes[j]
            
            # 1. Trouver TOUS les plus courts chemins
            try:
                # all_shortest_paths renvoie un générateur de chemins [s, ..., t]
                paths = list(nx.all_shortest_paths(G, source=s, target=t))
                num_total_paths = len(paths)
                
                # 2. Pour chaque noeud v du graphe (sauf s et t)
                for v in nodes:
                    if v == s or v == t:
                        continue
                        
                    # 3. Compter combien de chemins passent par v
                    paths_through_v = 0
                    for path in paths:
                        if v in path:
                            paths_through_v += 1
                    
                    # 4. Ajouter le ratio à la centralité de v
                    if num_total_paths > 0:
                        ratio = paths_through_v / num_total_paths
                        betweenness[v] += ratio
                        
            except nx.NetworkXNoPath:
                continue # Pas de chemin, on ignore
                
    return betweenness

bet_centrality = nx.betweenness_centrality(G_tokens, normalized=True, weight=None)

# Mise en forme des résultats ---
df_bet = pd.DataFrame(list(bet_centrality.items()), columns=['Token', 'Betweenness'])
df_bet = df_bet.sort_values('Betweenness', ascending=False)

# Analyse et Affichage ---
print("\n--- TOP 10 'PASSEURS DE SENS' (High Betweenness) ---")
print("Ces mots font le pont entre des communautés sémantiques distinctes.")
print(df_bet.head(10))

#Détection de Communauté pour contextualiser les ponts

token_frequencies = np.asarray(A_sparse.sum(axis=0)).flatten() #Sum the columns of the binary matrix
# token_frequencies is a array, we map it with the vocabulary
token_freq_dict = {vocabulary[i] : int(freq) for i, freq in enumerate(token_frequencies)}

communities = nx.community.louvain_communities(G_tokens, seed=42)

min_community_size = 10

significant_communities = [c for c in communities if len(c) >= min_community_size]
tiny_communities = [c for c in communities if len(c) < min_community_size]

print(f"Total communautés détectées : {len(communities)}")
print(f" -> Dont significatives (>= {min_community_size} tokens) : {len(significant_communities)}")
print(f" -> Dont micro-clusters (rejetées) : {len(tiny_communities)}")


# On crée un mapping Token -> ID Communauté
token_comm = {}
community_colors ={}

# Palette de couleurs hexadécimales pour l'export (générées aléatoirement ou fixes)
colors = ["#FF5733", "#33FF57", "#3357FF", "#F033FF", "#FF33A8", "#33FFF5", "#F5FF33"]

for i, comm in enumerate(significant_communities):
    for token in comm:
        token_comm[token] = i
        # On cherche les "Leaders" de la communauté (ceux avec le plus haut degré interne)
    subgraph = G_tokens.subgraph(comm)
    leaders = sorted(subgraph.degree, key=lambda x: x[1], reverse=True)[:5]
    leader_words = [word for word, deg in leaders]
        
    # Couleur (cycle sur la palette)
    color = colors[i % len(colors)]
    community_colors[i] = color
        
    print(f"Communauté {i} ({len(comm)} tokens) - Thème probable : {leader_words}")
    print(f"Exemples : {list(comm)[:8]}...")

# --- Enrichissement du Graphe pour Gephi ---

for node in G_tokens.nodes():
    # Attribut 1 : Modularity Class (Communauté) - Entier
    G_tokens.nodes[node]['modularity_class'] = token_comm.get(node, -1)
    
    # Attribut 2 : Frequency (Taille) - Entier
    freq = token_freq_dict.get(node, 1)
    G_tokens.nodes[node]['frequency'] = freq


# --- Export Gephi (.gexf) ---
output_file = "graphe_semantique_final.gexf"
nx.write_gexf(G_tokens, output_file)
print(f"\nExport terminé : {output_file}")

# Ajout de l'info communauté au tableau
df_bet['Community_ID'] = df_bet['Token'].map(token_comm)

print("\n--- ANALYSE CONTEXTUELLE DU TOP 5 ---")
top_5 = df_bet.head(5)['Token'].tolist()

for token in top_5:
    neighbors = list(G_tokens.neighbors(token))
    # On regarde les communautés des voisins
    neighbor_comms = [token_comm.get(n) for n in neighbors]
    unique_comms = set(neighbor_comms)
    
    print(f"\n🔹 Token : '{token}' (Score: {bet_centrality[token]:.4f})")
    print(f"   - Appartient à la Communauté : {token_comm.get(token)}")
    print(f"   - Connecté à {len(neighbors)} voisins")
    print(f"   - Fait le pont vers {len(unique_comms)} communautés différentes : {unique_comms}")


eigenvector = nx.eigenvector_centrality(G_tokens, max_iter=1000)

top10 = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:10]

print("\n--- Top 10 Mainstream (Eigenvector) ---")
for token, score in top10:
    print(token, ":", score)
