import json
import networkx as nx
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import os

'''
# CONFIGURATION
file_path = 'DATA/CLEAN/JSON/university_processed_features_qs.json'
top_n = 100               # Keep only the top 100 tokens per university (reduces noise)
JACCARD_THRESHOLD = 0.2   # CRITICAL THRESHOLD: If similarity < 20%, do not create a link

# ==============================================================================
# 1. DATA LOADING AND PREPARATION

if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found.")
else:
    with open(file_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

# Retrieve the dictionary {University: [list_of_tokens]}
raw_tokens_dict = full_data.get('tokens', {})

tokens_list = []
university_names = []

# Loop to extract and slice word lists
for name, tokens in raw_tokens_dict.items():
    university_names.append(name)
    tokens_list.append(tokens[:top_n]) # Slice at top_n to lighten computation

print(f"Number of universities extracted: {len(university_names)}")

# ==============================================================================
# 2. CALCULATING THE SIMILARITY MATRIX (JACCARD)

def create_jaccard_matrix(tokens_list):
    """
    Computes the Jaccard similarity matrix via matrix calculation (very fast).
    Jaccard(A, B) = (A inter B) / (A union B)
    """
    # 1. Transform into binary matrix (Rows=Univs, Cols=Words)
    # 1 if the word is present in the univ, 0 otherwise
    mlb = MultiLabelBinarizer(sparse_output=True)
    A_sparse = mlb.fit_transform(tokens_list)
    vocab = mlb.classes_ # The list of all unique words (the vocabulary)

    print(f"Matrix A (Univ x Tokens): {A_sparse.shape}")
    
    # 2. Transpose to get (Rows=Words, Cols=Univs)
    X = A_sparse.T 
    
    # 3. Calculate Intersection (Numerator)
    # Matrix multiplication: counts how many univs have both words in common
    intersection = (X * X.T).toarray()
    
    # 4. Calculate Union (Denominator)
    # First, count how many times each word appears in total (Frequency)
    token_degrees = np.array(X.sum(axis=1)).flatten()
    
    # Mathematical trick: Union(A,B) = Freq(A) + Freq(B) - Intersection(A,B)
    deg_matrix = np.add.outer(token_degrees, token_degrees)
    union_matrix = deg_matrix - intersection
    
    # 5. Final Division (Intersection / Union)
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccard_sim = intersection / union_matrix
        jaccard_sim = np.nan_to_num(jaccard_sim) # Replace errors (division by 0) with 0
    
    # Set diagonal to 0 (a word is not "similar" to itself in the graph context)
    np.fill_diagonal(jaccard_sim, 0)
    
    print(f"-> Jaccard Matrix computed: {jaccard_sim.shape}")
    return A_sparse, jaccard_sim, vocab, token_degrees

# Execute calculation
A_sparse, jaccard_sim, vocabulary, token_frequencies_array = create_jaccard_matrix(tokens_list)

# Create a dictionary {Word: Frequency} needed to orient arrows later
token_counts = {vocabulary[i]: int(freq) for i, freq in enumerate(token_frequencies_array)}

# ==============================================================================
# 3. CREATING THE DIRECTED GRAPH (THE COMPASS RULE)

def build_asymmetric_knn_graph(jaccard_matrix, vocab, token_counts_dict, k=5, threshold=0.0):
    """
    Builds the graph according to the logic: Specific (Rare) -> General (Frequent)
    """
    print(f"Building directed graph (k={k}, threshold > {threshold})...")
    DG = nx.DiGraph()
    num_tokens = len(vocab)

    for i in range(num_tokens):
        word_a = vocab[i]
        
        # 1. Find the k nearest neighbors (those with the highest Jaccard score)
        # argsort sorts in ascending order, we take the last ones [-k:]
        neighbor_indices = np.argsort(jaccard_matrix[i])[-k:]

        for j in neighbor_indices:
            word_b = vocab[j]
            weight = jaccard_matrix[i, j]

            # --- FILTER 1: THE THRESHOLD ---
            # If similarity is too low, ignore this link (it's noise)
            if weight < threshold: 
                continue 
            # -------------------------------

            # --- FILTER 2: ORIENTATION (COMPASS) ---
            # Retrieve global frequency of each word
            freq_a = token_counts_dict.get(word_a, 0)
            freq_b = token_counts_dict.get(word_b, 0)

            # Rule: The less frequent word points to the more frequent one
            # Ex: "Quantum" (Rare) -> "Physics" (Frequent)
            if freq_a < freq_b:
                DG.add_edge(word_a, word_b, weight=weight)
            elif freq_a > freq_b:
                DG.add_edge(word_b, word_a, weight=weight)
            else:
                # If frequencies are equal, use alphabetical order to decide cleanly
                if word_a < word_b:
                    DG.add_edge(word_a, word_b, weight=weight)
                else:
                    DG.add_edge(word_b, word_a, weight=weight)
    
    print(f"-> Directed Graph created: {DG.number_of_nodes()} nodes, {DG.number_of_edges()} edges.")
    return DG

# Generating the Graph with your parameters
G_directed = build_asymmetric_knn_graph(
    jaccard_sim, 
    vocabulary, 
    token_counts, 
    k=5, 
    threshold=JACCARD_THRESHOLD
)

# ==============================================================================
# 4. CLEANING: GIANT COMPONENT (MAINSTREAM)
# 

# Keep only the largest connected chunk of the graph.
# We use 'weakly_connected' because it is a directed graph (ignoring arrow direction for connection).
giant_component_nodes = max(nx.weakly_connected_components(G_directed), key=len)

# Create a subgraph containing only these nodes
G_main = G_directed.subgraph(giant_component_nodes).copy()

print(f"Mainstream Graph created: {G_main.number_of_nodes()} nodes, {G_main.number_of_edges()} edges.")

# ==============================================================================
# 5. EXPORT
# ==============================================================================
# This is the file you will open in Gephi
nx.write_gexf(G_main, "text_mining_applications/graph_dirige_tokens.gexf")
print("Graph exported successfully.")


'''

import json
import networkx as nx
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
file_path = 'DATA/CLEAN/JSON/university_processed_features_qs.json'
top_n = 100               
JACCARD_THRESHOLD = 0.2   
CONDITIONAL_THRESHOLD = 0.3 # NOUVEAU : On veut que A implique B au moins 30% du temps

# ==============================================================================
# 1. CHARGEMENT
# ==============================================================================
# (Identique à votre code précédent, je raccourcis pour la lisibilité)
if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found.")
    full_data = {}
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
# 2. CALCUL MATRICIEL OPTIMISÉ
# ==============================================================================

def create_advanced_matrices(tokens_list):
    # 1. Matrice Binaire
    mlb = MultiLabelBinarizer(sparse_output=True)
    A_sparse = mlb.fit_transform(tokens_list)
    vocab = mlb.classes_

    print(f"Matrix A (Univ x Tokens): {A_sparse.shape}")
    
    # 2. Co-occurrences (Intersection)
    X = A_sparse.T 
    intersection = (X * X.T).toarray() # Intersection(A, B)
    
    # 3. Fréquences individuelles
    token_degrees = np.array(X.sum(axis=1)).flatten()
    
    # 4. Calcul Jaccard (Pour le filtrage initial)
    deg_matrix = np.add.outer(token_degrees, token_degrees)
    union_matrix = deg_matrix - intersection
    
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccard_sim = intersection / union_matrix
        jaccard_sim = np.nan_to_num(jaccard_sim)
    np.fill_diagonal(jaccard_sim, 0)
    
    # 5. AMÉLIORATION : Calcul de la Probabilité Conditionnelle P(B|A)
    # P(B|A) = Intersection(A,B) / Freq(A)
    # Cela crée une matrice asymétrique !
    
    # On prépare la division par colonne (broadcasting)
    # On divise chaque ligne d'intersection par la fréquence du mot de la ligne
    with np.errstate(divide='ignore', invalid='ignore'):
        # Attention : token_degrees[:, None] permet de diviser ligne par ligne
        conditional_prob = intersection / token_degrees[:, None]
        conditional_prob = np.nan_to_num(conditional_prob)
        
    np.fill_diagonal(conditional_prob, 0)

    return jaccard_sim, conditional_prob, vocab, token_degrees

# Exécution
jaccard_sim, conditional_matrix, vocabulary, token_frequencies_array = create_advanced_matrices(tokens_list)
token_counts = {vocabulary[i]: int(freq) for i, freq in enumerate(token_frequencies_array)}

# ==============================================================================
# 3. CONSTRUCTION DU GRAPHE DIRIGÉ (PROBABILISTE)
# ==============================================================================

def build_smart_directed_graph(jaccard_mat, cond_mat, vocab, token_counts, k=5, jac_thresh=0.2):
    print(f"Building smart graph (k={k})...")
    DG = nx.DiGraph()
    num_tokens = len(vocab)

    for i in range(num_tokens):
        word_a = vocab[i]
        freq_a = token_counts.get(word_a, 0)
        
        # On utilise Jaccard pour sélectionner les voisins (car c'est une bonne mesure de proximité globale)
        neighbor_indices = np.argsort(jaccard_mat[i])[-k:]

        for j in neighbor_indices:
            word_b = vocab[j]
            freq_b = token_counts.get(word_b, 0)
            
            jac_score = jaccard_mat[i, j]

            # Filtre de base
            if jac_score < jac_thresh: continue 

            # --- AMÉLIORATION : DIRECTION LOGIQUE ---
            # On garde votre règle de la boussole (Rare -> Fréquent)
            # MAIS on utilise la probabilité conditionnelle comme poids !
            
            if freq_a < freq_b:
                # A pointe vers B
                # Le poids est P(B | A) : "A quel point A implique B ?"
                weight = cond_mat[i, j] # Probabilité que B soit là sachant A
                DG.add_edge(word_a, word_b, weight=weight, type='semantic_flow')
                
            elif freq_a > freq_b:
                # B pointe vers A
                # Le poids est P(A | B)
                weight = cond_mat[j, i]
                DG.add_edge(word_b, word_a, weight=weight, type='semantic_flow')
                
            else:
                # Égalité
                if word_a < word_b:
                    DG.add_edge(word_a, word_b, weight=cond_mat[i, j], type='semantic_flow')
                else:
                    DG.add_edge(word_b, word_a, weight=cond_mat[j, i], type='semantic_flow')

    print(f"-> Graph created: {DG.number_of_nodes()} nodes, {DG.number_of_edges()} edges.")
    return DG

G_directed = build_smart_directed_graph(
    jaccard_sim, 
    conditional_matrix,
    vocabulary, 
    token_counts, 
    k=5, 
    jac_thresh=JACCARD_THRESHOLD
)

# ==============================================================================
# 4. NETTOYAGE ET ENRICHISSEMENT
# ==============================================================================

# A. Composante Géante
giant_component_nodes = max(nx.weakly_connected_components(G_directed), key=len)
G_main = G_directed.subgraph(giant_component_nodes).copy()

# B. AMÉLIORATION : AJOUT D'ATTRIBUTS POUR GEPHI
nx.set_node_attributes(G_main, token_counts, 'frequency')
weighted_degree = dict(G_main.degree(weight='weight'))
nx.set_node_attributes(G_main, weighted_degree, 'weighted_degree')

# --- MODIFICATION ICI ---
print(f"Mainstream Graph created: {G_main.number_of_nodes()} nodes, {G_main.number_of_edges()} edges.")
# ------------------------