import json
import networkx as nx
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import os
from scipy.stats import entropy  # entropy helps identify context diversity
# High entropy: generic tokens like 'research', etc.
# Low entropy: specific tokens

# CONFIGURATION
file_path = 'DATA/CLEAN/JSON/university_processed_features_qs.json'
top_n = 100               
JACCARD_THRESHOLD = 0.2
K_NEIGHBORS = 5      


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

# Jaccard calculation

def create_jaccard_matrix(tokens_list):
    # 1. Binary Matrix
    mlb = MultiLabelBinarizer(sparse_output=True)
    A_sparse = mlb.fit_transform(tokens_list)
    vocab = mlb.classes_

    print(f"Matrix A (Univ x Tokens) shape: {A_sparse.shape}")
    
    # 2. Co-occurrences (Intersection)
    X = A_sparse.T 
    intersection = (X * X.T).toarray() 
    
    # 3. Individual Frequencies
    token_degrees = np.array(X.sum(axis=1)).flatten()
    
    # 4. Jaccard Calculation: Intersection / (Deg_i + Deg_j - Intersection)
    deg_matrix = np.add.outer(token_degrees, token_degrees)
    union_matrix = deg_matrix - intersection
    
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccard_sim = intersection / union_matrix
        jaccard_sim = np.nan_to_num(jaccard_sim)
        
    np.fill_diagonal(jaccard_sim, 0)
    
    return jaccard_sim, vocab, token_degrees, A_sparse

# Execution
jaccard_matrix, vocabulary, token_frequencies_array, A_sparse = create_jaccard_matrix(tokens_list)
token_counts = {vocabulary[i]: int(freq) for i, freq in enumerate(token_frequencies_array)}

# ENTROPY CALCULATION

def calculate_token_entropy(dtm_sparse, vocabulary):
    """
    Calculates the entropy of each token (measure of context diversity).
    - High entropy = Generic token (present across many different contexts/universities)
    - Low entropy = Specific token (concentrated in few universities)
    """
    token_entropy = {}
    
    for i, token in enumerate(vocabulary):
        # Binary vector: which universities use this token?
        token_vector = dtm_sparse[:, i].toarray().flatten()
        
        # Shannon Entropy normalized (log base doesn't matter for relative ranking)
        # Adding epsilon to avoid log(0)
        ent = entropy(token_vector + 1e-10)
        token_entropy[token] = ent
    
    return token_entropy

token_entropy_dict = calculate_token_entropy(A_sparse, vocabulary)

# Displaying generic and specific tokens
print("\n" + "=" * 60)
print("ENTROPY ANALYSIS (Genericity vs. Specificity)")
print("=" * 60)

sorted_by_entropy = sorted(token_entropy_dict.items(), key=lambda x: x[1], reverse=True)
print("\n TOP 10 GENERIC Tokens (High Entropy)")
for rank, (token, ent) in enumerate(sorted_by_entropy[:10], 1):
    print(f"{rank}. {token:<25} Entropy: {ent:.4f}  Frequency: {token_counts[token]}")

print("\n TOP 10 SPECIFIC Tokens (Low Entropy)")
for rank, (token, ent) in enumerate(sorted_by_entropy[-10:], 1):
    print(f"{rank}. {token:<25} Entropy: {ent:.4f}  Frequency: {token_counts[token]}")


# DIRECTED GRAPH CONSTRUCTION 

def build_directed_jaccard_graph_entropy(jac_mat, vocab, token_counts, token_entropy, k=10, thresh=0.15):
    """
    Builds a directed graph using Entropy as a directional compass.
    Direction: Generic (High Entropy) -> Specific (Low Entropy)
    """
    print(f"\nBuilding directed graph using ENTROPY compass...")
    print(f"(Metric=Jaccard, k={k}, threshold={thresh})")
    DG = nx.DiGraph()
    num_tokens = len(vocab)

    edges_added = 0
    for i in range(num_tokens):
        word_a = vocab[i]
        entropy_a = token_entropy.get(word_a, 0)
        
        # Look only at the top k neighbors according to Jaccard
        neighbor_indices = np.argsort(jac_mat[i])[-k:]

        for j in neighbor_indices:
            word_b = vocab[j]
            entropy_b = token_entropy.get(word_b, 0)
            
            score = jac_mat[i, j]
            if score < thresh: 
                continue 

            # ===== COMPASS RULE (ENTROPY) =====
            # Direction: Generic (High Entropy) -> Specific (Low Entropy)
            
            if entropy_a > entropy_b:
                # word_a is more generic than word_b
                DG.add_edge(word_a, word_b, weight=score, type='generic_to_specific')
                edges_added += 1
                
            elif entropy_a < entropy_b:
                # word_b is more generic than word_a
                DG.add_edge(word_b, word_a, weight=score, type='generic_to_specific')
                edges_added += 1
                
            else:
                # Equality case (rare), use alphabetical order to break symmetry
                if word_a < word_b:
                    DG.add_edge(word_a, word_b, weight=score, type='generic_to_specific')
                else:
                    DG.add_edge(word_b, word_a, weight=score, type='generic_to_specific')
                edges_added += 1

    print(f"-> Graph created: {DG.number_of_nodes()} nodes, {DG.number_of_edges()} edges.")
    return DG


G_directed = build_directed_jaccard_graph_entropy(
    jaccard_matrix, 
    vocabulary, 
    token_counts, 
    token_entropy_dict,
    k=K_NEIGHBORS, 
    thresh=JACCARD_THRESHOLD
)


# CLEANING (GIANT COMPONENT)

# Keep only the largest weakly connected component for consistent analysis
giant_component_nodes = max(nx.weakly_connected_components(G_directed), key=len)
G_main = G_directed.subgraph(giant_component_nodes).copy()

# Adding Node Attributes
nx.set_node_attributes(G_main, token_counts, 'frequency')
weighted_degree = dict(G_main.degree(weight='weight'))
nx.set_node_attributes(G_main, weighted_degree, 'weighted_degree')
nx.set_node_attributes(G_main, token_entropy_dict, 'entropy')

print("-" * 30)
print(f"FINAL NODE COUNT (G_main): {G_main.number_of_nodes()}")
print(f"FINAL EDGE COUNT (G_main): {G_main.number_of_edges()}")
print("-" * 30)

# PAGERANK & WEIGHTED HITS CALCULATION
print("Computing Prestige Metrics...")

#  PAGERANK 
try:
    pagerank_scores = nx.pagerank(G_main, alpha=0.85, weight='weight', max_iter=100, tol=1e-08)
    nx.set_node_attributes(G_main, pagerank_scores, 'pagerank')
    print("-> PageRank computed.")
except Exception as e:
    print(f"-> PageRank failed: {e}")

#  WEIGHTED HITS
def weighted_hits_numpy(G, weight_attr='weight', max_iter=100, tol=1e-08):
    nodes = list(G.nodes())
    n = len(nodes)
    M = nx.to_numpy_array(G, nodelist=nodes, weight=weight_attr)
    
    hubs = np.ones(n)
    auths = np.ones(n)
    
    for _ in range(max_iter):
        hubs_prev = hubs.copy()
        auths_prev = auths.copy()
        
        # Auth = Sum of Incoming Hubs * Weight (Jaccard)
        auths = np.dot(M.T, hubs)
        auths /= np.linalg.norm(auths)
        
        # Hub = Sum of Outgoing Auths * Weight (Jaccard)
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

# 6. RESULTS DISPLAY AND EXPORT

def print_top_k(scores_dict, metric_name, k=10):
    if not scores_dict: return
    sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    print(f"\n TOP {k} {metric_name.upper()}")
    print("-" * 40)
    for rank, (node, score) in enumerate(sorted_items, 1):
        print(f"{rank}. {node:<25} : {score:.5f}")

print("=" * 60)
print("RESULTS SUMMARY (JACCARD + ENTROPY)")
print("=" * 60)

print_top_k(pagerank_scores, "PageRank (Influence)")
print_top_k(authorities_scores, "HITS Authorities (Key Concepts)")
print_top_k(hubs_scores, "HITS Hubs (Best Connectors)")

# EXPORT
print("\n" + "=" * 60)
print(f"Exporting Giant Component to Gephi format...")
output_filename = "GEXF/final_directed_graph.gexf"
nx.write_gexf(G_main, output_filename)
print(f"Done! File saved as '{output_filename}'.")
print("=" * 60)