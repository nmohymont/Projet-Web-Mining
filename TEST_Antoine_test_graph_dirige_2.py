import json
import networkx as nx
import os

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# Chemin vers votre fichier JSON
FILE_PATH = 'DATA/CLEAN/JSON/university_processed_features_qs.json'
# FILE_PATH = 'university_processed_features_qs.json' 

# Nombre de mots à garder par université (Top N)
TOP_N_TOKENS = 50

# ==============================================================================
# 2. CHARGEMENT
# ==============================================================================

def load_data_pairs(filepath, top_n):
    if not os.path.exists(filepath):
        print(f"ERREUR : Le fichier {filepath} est introuvable.")
        return []

    print(f"Chargement de {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    raw_tokens = data.get('tokens', {})
    
    univ_token_pairs = []
    for name, tokens in raw_tokens.items():
        if tokens:
            univ_token_pairs.append((name.strip(), tokens[:top_n]))
            
    print(f"-> {len(univ_token_pairs)} universités chargées.")
    return univ_token_pairs

# ==============================================================================
# 3. CRÉATION DU GRAPHE BIPARTITE DIRIGÉ
# ==============================================================================

def create_bipartite_graph(data_pairs):
    print("\nConstruction du Graphe Bipartite (Univ -> Tokens)...")
    G = nx.DiGraph()
    edge_list = []
    
    for univ_name, tokens in data_pairs:
        # Ajout du Nœud Université (Source)
        G.add_node(univ_name, type='University', label=univ_name)
        
        for token in tokens:
            # Ajout du Nœud Token (Cible)
            G.add_node(token, type='Token', label=token)
            # Lien dirigé : Univ -> Token
            edge_list.append((univ_name, token))
            
    G.add_edges_from(edge_list)
    print(f"-> Graphe créé : {G.number_of_nodes()} nœuds ({len(edge_list)} liens).")
    return G

# ==============================================================================
# 4. EXÉCUTION ET ANALYSE
# ==============================================================================

data = load_data_pairs(FILE_PATH, TOP_N_TOKENS)

if data:
    G = create_bipartite_graph(data)
    
    print("\nCalcul des métriques (HITS, Degree, PageRank)...")
    
    # 1. HITS
    try:
        hubs, authorities = nx.hits(G, max_iter=100)
    except:
        hubs, authorities = {}, {}
        print("Erreur HITS (non-convergence)")

    # 2. Degrees
    out_degree = dict(G.out_degree())
    in_degree = dict(G.in_degree())

    # 3. PageRank
    try:
        pagerank = nx.pagerank(G, alpha=0.85)
    except:
        pagerank = {n: 0 for n in G.nodes()}
        print("Erreur PageRank")

    # --- Affichage Console ---
    print("\n" + "="*60)
    print("RÉSULTATS STATISTIQUES")
    print("="*60)
    
    # Top PageRank (Dominé par les Tokens)
    print("\n--- TOP 20 PAGERANK (Concepts les plus centraux) ---")
    top_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:20]
    for n, score in top_pr:
        n_type = G.nodes[n].get('type')
        print(f"   - [{n_type}] {n} (PR: {score:.4f})")

    # Top Hubs (Universités les plus représentatives)
    print("\n--- TOP 20 HUBS (Universités couvrant le mieux les concepts) ---")
    # On filtre pour ne garder que les universités
    top_hubs = sorted(
        [(n, hubs.get(n,0)) for n in G.nodes if G.nodes[n].get('type') == 'University'],
        key=lambda x: x[1], reverse=True
    )[:20]
    for n, score in top_hubs:
        print(f"   - {n} (Hub Score: {score:.4f})")

    # --- Export Gephi ---
    output_file = "graphe_bipartite_univ_token.gexf"
    print(f"\nExportation vers {output_file}...")
    
    # Intégration des scores dans le fichier GEXF pour utilisation dans Gephi
    nx.set_node_attributes(G, hubs, 'hub_score')
    nx.set_node_attributes(G, authorities, 'authority_score')
    nx.set_node_attributes(G, in_degree, 'in_degree')
    nx.set_node_attributes(G, pagerank, 'pagerank_score')
    
    nx.write_gexf(G, output_file)
    print("Export terminé avec succès.")

else:
    print("Aucune donnée chargée.")


    