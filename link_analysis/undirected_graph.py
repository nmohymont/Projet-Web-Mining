import json
import pandas as pd
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from adjust_text import adjust_text # Pour le décalage automatique

from sklearn.preprocessing import MultiLabelBinarizer 
#instead of for loop based on the token stored in the JSON file per university usde MultiLabelBinazrier 

def degree_matrix(A: np.ndarray, direction: str = "out") -> np.ndarray:
    """
    Compute the degree matrix (either in-degree or out-degree).

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (n x n)
    direction : str, optional
        'out' for out-degree (default)
        'in'  for in-degree

    Returns
    -------
    D : np.ndarray
        Diagonal degree matrix
    """

    n = A.shape[0] # number of nodes
    D = np.zeros((n, n), dtype=int) # Initialize degree matrix size n x n

    if direction == "out": # Compute out-degrees
        degrees = np.sum(A, axis=1) # Sum rows
    elif direction == "in": # Compute in-degrees
        degrees = np.sum(A, axis=0) # Sum columns
    else:
        raise ValueError("Direction must be 'out' or 'in'.")

    np.fill_diagonal(D, degrees) # Fill diagonal with degrees or np.diag(degrees) juste avec les degrees et pas créer la matrice complète de 0.
    return D

def shortest_path_matrix(A: np.ndarray) -> np.ndarray:
    """
    Compute the all-pairs shortest path matrix from an adjacency matrix A
    using Floyd–Warshall, with the same logic as your preprocess() + floyd().

    - A is assumed to be an unweighted adjacency matrix (0 = no edge, 1 = edge)
    - unreachable pairs will get a large value (100000)
    """
    # work on a copy so we don't modify A
    SP = A.copy().astype(float)
    n = SP.shape[0]

    # --- preprocess step: replace 0 by "infinity"
    for i in range(n):
        for j in range(n):
            if SP[i, j] == 0:
                SP[i, j] = 100000.0

    # Floyd–Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                SP[i, j] = min(SP[i, j], SP[i, k] + SP[k, j])

    # distance from a node to itself is 0
    for i in range(n):
        SP[i, i] = 0.0

    return SP

def closeness_centrality(SP: np.ndarray) -> np.ndarray:
    """
    Standard closeness centrality:
        C(i) = (n - 1) / Σ_j d(i, j)
    Computed using the shortest path matrix.
    """

    n = SP.shape[0]
    sum_shortest_path= np.sum(SP, axis=1)
    
    # On évite la division par zéro si un nœud est isolé
    with np.errstate(divide='ignore', invalid='ignore'):
        closeness = (n - 1) / sum_shortest_path
        closeness = np.nan_to_num(closeness)
    return closeness

def residual_closeness_centrality(A: np.ndarray) -> np.ndarray: # need to run shortest path every time it delete a node
    """
    Residual closeness centrality:
        RCC(k) = Σ_{i ≠ j} (1 / 2^{d(i, j)})
    Computed by removing node k from the graph, recalculating
    the shortest paths on the remaining subgraph, and summing
    the inverse powers of the distances.
    """
    n = A.shape[0] # nombre de noeud
    #sum_shortest_path = np.sum(shortest_path_matrix(A), axis=1)

    rc = np.zeros(n)
    
    for k in range(n):
        #print( f"Removing node {k}:")
        A_temp = np.delete(A, k, axis=0)  # remove row k 
        A_rc = np.delete(A_temp, k, axis=1) # remove column k
        #print(A_rc)
        #print(shortest_path_matrix(A_rc))
        for i in range(n-1):
            for j in range(n-1):
                if i != j:
                    d_ij = shortest_path_matrix(A_rc)[i, j]
                    rc[k] += 1 / (2 ** d_ij)
    return rc

def eccentricity_centrality(A: np.ndarray) -> np.ndarray:
    """
    Eccentricity centrality of each node:
        ecc(i) = 1 / max_j d(i, j)
    The inverse is used so that larger distances correspond
    to smaller eccentricity values. Unreachable distances
    (set to 100000) are ignored in the logic of shortest_path_matrix.
    """
    ecc = np.zeros(A.shape[0])
    SP = shortest_path_matrix(A) # récupère la matrice des plus courts chemins entre chaque noeuds
    for i in range(SP.shape[0]): # on marque chaque ligne i 
        max_distance = np.max(SP[i, :]) # on trouve le maximum dans toutes les colonnes de la ligne i
        ecc_i = 1 / max_distance # on calcule l'excentricité
        ecc[i] = ecc_i #on stocke le résultat dans le tableau ecc
    return ecc

def common_neighbors_matrix(A: np.ndarray) -> np.ndarray: # measure co-occurence
    CN = A @ A.T

    #in loop
    #CN = np.zeros(A.shape, dtype=float)
    #n = A.shape[0]
    #for i in range(n):
    #    for k in range(n): # Iterate over columns
            #prod = 0.0
            #for j in range(n):  # Iterate over rows
             #   prod += A[i, j] * A[k, j]
    #       CN[i, j] = prod 
    return CN

def katz_matrix(A: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Katz = (I - alpha A)^(-1) - I


    """
    n = A.shape[0]
    I = np.eye(n)  # Identity matrix
    Katz = np.linalg.inv(I - alpha * A) - I
    return Katz

def create_undirected_graph_jaccard(tokens_list):
    """
    Computes the Jaccard similarity matrix for tokens based on their 
    co-occurrence across universities.
    
    Formula:
        J(i, j) = |Intersection(i, j)| / (|Degree(i)| + |Degree(j)| - |Intersection(i, j)|)
    
    Returns:
        A_sparse : Sparse document-term matrix (Universities x Tokens)
        jaccard_sim : Dense Jaccard similarity matrix
        vocab : Array of token names corresponding to matrix indices
    """
    # 1. Create the Document-Term Matrix (binary)
    # This is the "A" matrix (Universities x Tokens)
    mlb = MultiLabelBinarizer(sparse_output=True)
    A_sparse = mlb.fit_transform(tokens_list)
    vocab = mlb.classes_

    print(f"Matrix A (Universities x Tokens): {A_sparse.shape}")
    
    # 2. Transpose to get Token-University Matrix (X)
    X = A_sparse.T  # Tokens x Universities

    print(f"Matrix X (Tokens x Universities): {X.shape}")
    
    # 3. Calculate Intersection (numerator)
    # Matrix multiplication: number of shared universities between token i and j
    intersection = (X * X.T).toarray()
    
    # 4. Calculate Degrees
    # Number of universities using each token
    token_degrees = np.array(X.sum(axis=1)).flatten()
    
    # 5. Calculate Union (denominator) using broadcasting
    # deg_matrix[i, j] = degree(i) + degree(j)
    deg_matrix = np.add.outer(token_degrees, token_degrees)
    union_matrix = deg_matrix - intersection
    
    # 6. Final Jaccard calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccard_sim = intersection / union_matrix
        # Handle cases where union is zero
        jaccard_sim = np.nan_to_num(jaccard_sim)
    
    # Remove self-similarity (diagonal) for network analysis
    np.fill_diagonal(jaccard_sim, 0)
    
    print(f"-> Jaccard matrix computed for {jaccard_sim.shape[0]} tokens.")
    return A_sparse, jaccard_sim, vocab


def plot_katz_vs_degree(df_analysis):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_analysis, x='Degree', y='Katz', alpha=0.6)
    
    # Annotate top 5 Katz scores
    top_katz = df_analysis.nlargest(5, 'Katz')
    for i, row in top_katz.iterrows():
        plt.text(row['Degree'], row['Katz'], row['Token'], fontsize=9, fontweight='bold')
    
    plt.title("Global Analysis: Volume (Degree) vs Prestige (Katz)")
    plt.xlabel("Number of Neighbors (Degree)")
    plt.ylabel("Cumulative Influence (Katz)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def compute_rcc_for_top_betweenness(A, nodes, top_tokens):
    """
    Computes RCC only for a specific list of tokens (Top Betweenness)
    to save computation time.
    """
    results = {}
    for token in top_tokens:
        k = nodes.index(token) # Get the index of the token
        
        # 1. Create subgraph by removing node k
        A_rc = np.delete(np.delete(A, k, axis=0), k, axis=1)
        
        # 2. Get shortest paths for the new subgraph
        SP_rc = shortest_path_matrix(A_rc)
        
        # 3. Calculate RCC formula
        # We only sum 1/2^d for finite distances (ignore our 100,000 constant)
        score = np.sum(1 / (2 ** SP_rc[SP_rc < 100000])) - (SP_rc.shape[0]) # -n to remove d(i,i)=0
        results[token] = score
        
        print(f"RCC for '{token}' computed.")
    return results

file_path = 'DATA/CLEAN/JSON/university_processed_features_qs.json'
top_n = 100

with open(file_path, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

# extract raw token from json file
raw_tokens_dict = full_data.get('tokens', {})

tokens_list = []
university_names = []

for name, tokens in raw_tokens_dict.items():
    university_names.append(name)
    # we extract the top_n best token based on TF-IDF
    tokens_list.append(tokens[:top_n]) 

print(f"Number of extracted universities: {len(university_names)}")
print(f"Example for {university_names[0]} : {tokens_list[0][:5]}")

A_sparse, jaccard_sim, vocabulary = create_undirected_graph_jaccard(tokens_list)
print("Similarity matrix done ")

# Graph Creation 
G_tokens = nx.Graph()
THRESHOLD = 0.20

# only consider token id jaccard_similarity above threshold
rows, cols = np.where(jaccard_sim > THRESHOLD)

edge_list = []
for i, j in zip(rows, cols):
    if i < j: # only upper triangle of the matrix 
        edge_list.append((vocabulary[i], vocabulary[j], jaccard_sim[i, j]))

G_tokens.add_weighted_edges_from(edge_list)

print(f"Graph created: {G_tokens.number_of_nodes()} nodes, {G_tokens.number_of_edges()} edges.")

# an export was made here to see what the graph looked like

# Before proceeding with metric measurements, it is necessary to visualize the formed clusters.
# We apply the Louvain algorithm to identify the communities that group only the tokens related to artificial and intelligence.

token_frequencies = np.asarray(A_sparse.sum(axis=0)).flatten() #Sum the columns of the binary matrix
# token_frequencies is a array, we map it with the vocabulary
token_freq_dict = {vocabulary[i] : int(freq) for i, freq in enumerate(token_frequencies)}

communities = nx.community.louvain_communities(G_tokens, seed=42)

min_community_size = 10

significant_communities = [c for c in communities if len(c) >= min_community_size]
tiny_communities = [c for c in communities if len(c) < min_community_size]

print(f"Total communities detected : {len(communities)}")
print(f" -> Significant (>= {min_community_size} tokens) : {len(significant_communities)}")
print(f" -> Micro-clusters (rejected) : {len(tiny_communities)}")

# mapping Token -> ID Communauté
token_comm = {}


for i, comm in enumerate(significant_communities):
    for token in comm:
        token_comm[token] = i
        # looking for the community "Leaders" (highest degree)
    subgraph = G_tokens.subgraph(comm)
    leaders = sorted(subgraph.degree, key=lambda x: x[1], reverse=True)[:5]
    leader_words = [word for word, deg in leaders]
        
    print(f"Community {i} ({len(comm)} tokens) - Probable theme : {leader_words}")
    print(f"Examples : {list(comm)[:8]}...")

# Beyond the Louvain algorithm, we can extract the giant component to isolate the dissociated micro-clusters
# This makes it possible to perform centrality analyses on the mainstream (giant component) as a single connected graph

# Extract the giant component ("mainstream")
giant_component_nodes = max(nx.connected_components(G_tokens), key=len)
G_main = G_tokens.subgraph(giant_component_nodes).copy()

print(f" Mainstream Graph created : {G_main.number_of_nodes()} nodes, {G_main.number_of_edges()} edges.")

# Convert the graph into a NumPy adjacency matrix
adj_matrix = nx.to_numpy_array(G_main, weight=None)
nodes = list(G_main.nodes())


D_out = degree_matrix(adj_matrix, direction="out")

degrees_list = np.diag(D_out)

top10_indices = np.argsort(degrees_list)[::-1][:10]

print("Top 10 des degrees :")
for idx in top10_indices:
    print(f"node {nodes[idx]} -> degree = {degrees_list[idx]}")

SP_matrix = shortest_path_matrix(adj_matrix)

closeness_vals = closeness_centrality(adj_matrix)

eccentricity_vals = eccentricity_centrality(adj_matrix)

cn_matrix = common_neighbors_matrix(adj_matrix)
np.fill_diagonal(cn_matrix,0)
ind_x, ind_y = np.unravel_index(np.argsort(cn_matrix, axis=None)[-10:], cn_matrix.shape)

katz_mat = katz_matrix(adj_matrix, alpha=0.01)
katz_vals = np.array(katz_mat.sum(axis=1)).flatten() # sum the rows for the score per nodes 

# Summarize in a DataFrame
df_analysis = pd.DataFrame({
    'Token': nodes,
    'Degree': degrees_list,
    'Closeness': closeness_vals,
    'Eccentricity': eccentricity_vals,
    'Katz': katz_vals
})

plot_katz_vs_degree(df_analysis)


print("--- Center ---")
max_ecc_cent = df_analysis['Eccentricity'].max() # We must take the maximum because the computed eccentricity is the centralized version whose inverse was taken
# The radius is a single value for the entire graph, unlike eccentricity which is defined for each node
radius_val = 1 / max_ecc_cent if max_ecc_cent != 0 else 0
print(f"Radius : {radius_val:.2f}")


center_tokens = df_analysis[df_analysis['Eccentricity'] == max_ecc_cent]['Token'].tolist()
print(f"Tokens at the Center : {center_tokens}")

print("\n--- Periphery ---")
min_ecc_cent = df_analysis['Eccentricity'].min()
diameter_val = 1 / min_ecc_cent if min_ecc_cent != 0 else 0
print(f"Diameter : {diameter_val:.2f}")

periphery_tokens = df_analysis[df_analysis['Eccentricity'] == min_ecc_cent]['Token'].tolist()
print(f"Periphery token : {periphery_tokens}")

print("\n--- Top Closeness - accessibility mesure   ---")
print(df_analysis.sort_values('Closeness', ascending=False).head(5))

print("\n--- Top Common Neighbors ---")
for i, j in zip(ind_x, ind_y):
    print(f"{nodes[i]} <-> {nodes[j]} (Communs Neighbors: {cn_matrix[i,j]})")


def freeman_betweenness(G):  # used to see what is used to calculate the betweenness
    """
    Literal implementation of Freeman's formula.
    Note: Slow for large graphs (O(N^3)).
    For production, use nx.betweenness_centrality (Brandes algorithm).
    """
    nodes = list(G.nodes())
    n = len(nodes)
    betweenness = {node: 0.0 for node in nodes}
    
    # For each unique pair (s, t)
    for i in range(n):
        for j in range(i + 1, n):
            s = nodes[i]
            t = nodes[j]
            
            # Find ALL shortest paths
            try:
                # all_shortest_paths returns a generator of paths [s, ..., t]
                paths = list(nx.all_shortest_paths(G, source=s, target=t))
                num_total_paths = len(paths)
                
                #  For each node v in the graph (except s and t)
                for v in nodes:
                    if v == s or v == t:
                        continue
                        
                    #  Count how many paths pass through v
                    paths_through_v = 0
                    for path in paths:
                        if v in path:
                            paths_through_v += 1
                    
                    #  Add the ratio to the centrality of v
                    if num_total_paths > 0:
                        ratio = paths_through_v / num_total_paths
                        betweenness[v] += ratio
                        
            except nx.NetworkXNoPath:
                continue  # No path, ignore
                
    return betweenness


bet_centrality = nx.betweenness_centrality(G_main, normalized=True, weight=None)


df_bet = pd.DataFrame(list(bet_centrality.items()), columns=['Token', 'Betweenness'])
df_bet = df_bet.sort_values('Betweenness', ascending=False)

print("\n--- TOP 10 High Betweenness ---")


# --- Identify Target Tokens ---
# Combine Top 10 Betweenness and Center tokens (removing duplicates)
top_10_bet = df_bet.head(10)['Token'].tolist()
print(top_10_bet)


print(f"\n--- Starting Residual closeness centrality Calculation for top10 Target Tokens ---")

rcc_result_dict = compute_rcc_for_top_betweenness(adj_matrix, nodes, top_10_bet)

print("\n" + "="*50)
print(f"{'Token':<20} | {'RCC Score':<15} ")
print("-"*50)

for token_name in top_10_bet:
    score = rcc_result_dict[token_name]
    
    print(f"{token_name:<20} | {score:<15.4f}")

# extra for Gephi

for node in G_main.nodes():
    # Attribut 1 : Modularity Class 
    G_main.nodes[node]['modularity_class'] = token_comm.get(node, -1)
    
    # Attribut 2 : Frequency (Taille) 
    freq = token_freq_dict.get(node, 1)
    G_main.nodes[node]['frequency'] = freq


# --- Export Gephi (.gexf) ---
output_file = "graphe_semantique_final.gexf"
nx.write_gexf(G_main, output_file)
print(f"\nExport completed : {output_file}")

df_bet['Community_ID'] = df_bet['Token'].map(token_comm)

print("\n--- CONTEXTUAL ANALYSIS OF TOP 5 BROKERS ---")
for token in df_bet.head(5)['Token']:
    neighbors = list(G_main.neighbors(token))
    unique_comms = set([token_comm.get(n) for n in neighbors if n in token_comm])
    print(f"🔹 Token: '{token}' | Connects {len(neighbors)} neighbors across {len(unique_comms)} communities.")



