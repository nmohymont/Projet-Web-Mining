import json
import networkx as nx
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import os


# CONFIGURATION
file_path = 'DATA/CLEAN/JSON/university_processed_features_qs.json'
top_n = 100               
JACCARD_THRESHOLD = 0.2
K_NEIGHBORS = 5      

# ==============================================================================
# 1. CHARGEMENT

if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found.")
    print("Please run the preprocessing script first to generate the JSON.")
    exit()
else:
    with open(file_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

raw_tokens_dict = full_data.get('tokens', {})
tokens_list = []
university_names = []

for name, tokens in raw_tokens_dict.items():
    university_names.append(name)
    tokens_list.append(tokens[:top_n])

print(f"Number of universities extracted: {len(university_names)}")

# ==============================================================================
# 2. CALCUL MATRICIEL (JACCARD)

def create_jaccard_matrix(tokens_list):
    # 1. Matrice Binaire
    mlb = MultiLabelBinarizer(sparse_output=True)
    A_sparse = mlb.fit_transform(tokens_list)
    vocab = mlb.classes_

    print(f"Matrix A (Univ x Tokens): {A_sparse.shape}")
    
    # 2. Co-occurrences (Intersection)
    X = A_sparse.T 
    intersection = (X * X.T).toarray() 
    
    # 3. Fréquences individuelles
    token_degrees = np.array(X.sum(axis=1)).flatten()
    
    # 4. Calcul Jaccard : Intersection / (Deg_i + Deg_j - Intersection)
    deg_matrix = np.add.outer(token_degrees, token_degrees)
    union_matrix = deg_matrix - intersection
    
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccard_sim = intersection / union_matrix
        jaccard_sim = np.nan_to_num(jaccard_sim)
        
    np.fill_diagonal(jaccard_sim, 0)
    
    return jaccard_sim, vocab, token_degrees

# Exécution
jaccard_matrix, vocabulary, token_frequencies_array = create_jaccard_matrix(tokens_list)
token_counts = {vocabulary[i]: int(freq) for i, freq in enumerate(token_frequencies_array)}

# ==============================================================================
# 3. CONSTRUCTION DU GRAPHE DIRIGÉ (JACCARD + BOUSSOLE)

def build_directed_jaccard_graph(jac_mat, vocab, token_counts, k=10, thresh=0.15):
    print(f"Building graph (Metric=Jaccard, k={k}, threshold={thresh})...")
    DG = nx.DiGraph()
    num_tokens = len(vocab)

    for i in range(num_tokens):
        word_a = vocab[i]
        freq_a = token_counts.get(word_a, 0)
        
        # On ne regarde que les k meilleurs amis selon Jaccard
        neighbor_indices = np.argsort(jac_mat[i])[-k:]

        for j in neighbor_indices:
            word_b = vocab[j]
            freq_b = token_counts.get(word_b, 0)
            
            score = jac_mat[i, j]

            # Filtre de qualité
            if score < thresh: continue 

            # RÈGLE DE LA BOUSSOLE (Rare -> Fréquent)
            # Le poids EST le score Jaccard (Force du lien)
            
            if freq_a < freq_b:
                DG.add_edge(word_a, word_b, weight=score, type='semantic_link')
                
            elif freq_a > freq_b:
                DG.add_edge(word_b, word_a, weight=score, type='semantic_link')
                
            else:
                # Cas d'égalité (rare), ordre alphabétique
                if word_a < word_b:
                    DG.add_edge(word_a, word_b, weight=score, type='semantic_link')
                else:
                    DG.add_edge(word_b, word_a, weight=score, type='semantic_link')

    print(f"-> Graph created: {DG.number_of_nodes()} nodes, {DG.number_of_edges()} edges.")
    return DG

G_directed = build_directed_jaccard_graph(
    jaccard_matrix, 
    vocabulary, 
    token_counts, 
    k=K_NEIGHBORS, 
    thresh=JACCARD_THRESHOLD
)

# ==============================================================================
# 4. NETTOYAGE (COMPOSANTE GÉANTE)

# On ne garde que la plus grosse partie connectée pour analyse cohérente
giant_component_nodes = max(nx.weakly_connected_components(G_directed), key=len)
G_main = G_directed.subgraph(giant_component_nodes).copy()

# Ajout des attributs
nx.set_node_attributes(G_main, token_counts, 'frequency')
weighted_degree = dict(G_main.degree(weight='weight'))
nx.set_node_attributes(G_main, weighted_degree, 'weighted_degree')

print(f"Mainstream Graph (Giant Component): {G_main.number_of_nodes()} nodes.")

# ==============================================================================
# 5. CALCULS DE CENTRALITÉ (PAGERANK & WEIGHTED HITS)
print("Computing Centrality Metrics...")

# --- A. PAGERANK ---
try:
    pagerank_scores = nx.pagerank(G_main, alpha=0.85, weight='weight', max_iter=100, tol=1e-08)
    nx.set_node_attributes(G_main, pagerank_scores, 'pagerank')
    print("-> PageRank computed.")
except Exception as e:
    print(f"-> PageRank failed: {e}")

# --- B. WEIGHTED HITS (MANUEL) ---
# Correspond à l'option "Use Edge Weights" de Gephi
def weighted_hits_numpy(G, weight_attr='weight', max_iter=100, tol=1e-08):
    nodes = list(G.nodes())
    n = len(nodes)
    M = nx.to_numpy_array(G, nodelist=nodes, weight=weight_attr)
    
    hubs = np.ones(n)
    auths = np.ones(n)
    
    for _ in range(max_iter):
        hubs_prev = hubs.copy()
        auths_prev = auths.copy()
        
        # Auth = Somme des Hubs entrants * Poids(Jaccard)
        auths = np.dot(M.T, hubs)
        auths /= np.linalg.norm(auths)
        
        # Hub = Somme des Auths sortants * Poids(Jaccard)
        hubs = np.dot(M, auths)
        hubs /= np.linalg.norm(hubs)
        
        if np.linalg.norm(hubs - hubs_prev) < tol and np.linalg.norm(auths - auths_prev) < tol:
            break
            
    return dict(zip(nodes, hubs)), dict(zip(nodes, auths))

try:
    hubs_scores, authorities_scores = weighted_hits_numpy(G_main, weight_attr='weight')
    nx.set_node_attributes(G_main, hubs_scores, 'hubs')
    nx.set_node_attributes(G_main, authorities_scores, 'authorities')
    print("-> Weighted HITS computed.")
except Exception as e:
    print(f"-> HITS failed: {e}")

# ==============================================================================
# 6. AFFICHAGE ET EXPORT

def print_top_k(scores_dict, metric_name, k=10):
    if not scores_dict: return
    sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    print(f"\n🏆 TOP {k} {metric_name.upper()}")
    print("-" * 40)
    for rank, (node, score) in enumerate(sorted_items, 1):
        print(f"{rank}. {node:<25} : {score:.5f}")

print("=" * 60)
print("RÉSULTATS (JACCARD + BOUSSOLE)")
print("=" * 60)

print_top_k(pagerank_scores, "PageRank (Influence)")
print_top_k(authorities_scores, "HITS Authorities (Concepts Clés)")
print_top_k(hubs_scores, "HITS Hubs (Meilleurs Connecteurs)")



#Export
print("\n" + "=" * 60)
print(f"Exporting Giant Component to Gephi format...")
output_filename = "university_semantic_graph_directed.gexf"
nx.write_gexf(G_main, output_filename)
print(f"Done! File saved as '{output_filename}'.")
print("=" * 60)