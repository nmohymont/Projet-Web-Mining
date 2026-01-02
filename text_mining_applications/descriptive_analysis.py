import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
from collections import Counter
import math

import json  # <--- REPLACES PICKLE
import os

import networkx as nx 
from itertools import combinations

import seaborn as sns

import os

import nltk
from nltk.text import Text


# ==============================================================================
# 1 - Word Clouds Before/After 2015 (100% JSON VERSION)


# --- FILE CONFIGURATION ---
files_config = {

    'pre_2015': [

        'DATA/CLEAN/JSON/donnees_traitees_the_2012.json'  # Before 2015

    ],

    'post_2015': [

        'DATA/CLEAN/JSON/donnees_traitees_qs.json',       # WARNING: I added the missing comma here

        'DATA/CLEAN/JSON/donnees_traitees_the.json',

        'DATA/CLEAN/JSON/donnees_traitees_the_2021.json'

    ]

}
# --- LOADING FUNCTION ---
def load_and_aggregate_tokens(file_list):
    """Loads multiple JSON files and combines all tokens."""
    aggregated_tokens = []
    
    for file_path in file_list:
        try:
            if not os.path.exists(file_path):
                # Just ignore silently or print small warning
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                docs_tokens = data.get("tokens", {})
                
                if not docs_tokens:
                    continue

                count_unis = 0
                for tokens_list in docs_tokens.values():
                    aggregated_tokens.extend(tokens_list)
                    count_unis += 1
                
                print(f"   -> Loaded : {os.path.basename(file_path)} ({count_unis} universities)")
                
        except Exception as e:
            print(f"   Error on {file_path} : {e}")
            
    return aggregated_tokens

# --- EXECUTION ---
print("=== LOADING AND AGGREGATION ===")

print("\n1. 'LEGACY' Corpus (Before 2015)...")
tokens_pre_2015 = load_and_aggregate_tokens(files_config['pre_2015'])

print("\n2. 'RESPONSIBILITY' Corpus (After 2015)...")
tokens_post_2015 = load_and_aggregate_tokens(files_config['post_2015'])

print(f"\nTotal words Before : {len(tokens_pre_2015)}")
print(f"Total words After : {len(tokens_post_2015)}")

# --- DISPLAY WORD COUNTS IN PROMPT ---
def print_top_words(tokens, title, top_n=30):
    print(f"\n>>> TOP {top_n} WORDS : {title} <<<")
    print("-" * 50)
    print(f"{'Rank':<5} | {'Word':<20} | {'Frequency':<10}")
    print("-" * 50)
    counts = Counter(tokens).most_common(top_n)
    for rank, (word, freq) in enumerate(counts, 1):
        print(f"{rank:<5} | {word:<20} | {freq:<10}")
    print("-" * 50)

# AFFICHE LES MOTS DANS LE PROMPT ICI :
print_top_words(tokens_pre_2015, "BEFORE 2015 (Old World)")
print_top_words(tokens_post_2015, "AFTER 2015 (New World)")

# --- VISUALIZATION (WORD CLOUDS) ---
def plot_compare_wordclouds(tokens1, tokens2, title1, title2):
    if not tokens1 or not tokens2:
        print("Insufficient data for word clouds.")
        return

    text1 = " ".join(tokens1)
    text2 = " ".join(tokens2)

    # Color choise 
    wc1 = WordCloud(width=800, height=500, background_color='white', 
                   collocations=False, max_words=60, 
                   colormap='Dark2').generate(text1) 
                   
    wc2 = WordCloud(width=800, height=500, background_color='white', 
                   collocations=False, max_words=60, 
                   colormap='winter').generate(text2) 

    # Display
    fig, axes = plt.subplots(1, 2, figsize=(22, 12))

    # Cloud 1
    axes[0].imshow(wc1, interpolation='bilinear')
    axes[0].set_title(f"{title1}\n(The Old World)", fontsize=18, fontweight='bold', color="#800080") 
    axes[0].axis('off')

    # Cloud 2
    axes[1].imshow(wc2, interpolation='bilinear')
    axes[1].set_title(f"{title2}\n(The New World)", fontsize=18, fontweight='bold', color='#00008B')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

print("\n=== GENERATING WORD CLOUDS ===")
plot_compare_wordclouds(tokens_pre_2015, tokens_post_2015, "Before 2015", "After 2015")

# ==============================================================================
# 2 - Word Clouds by Continent (OPTIMIZED DISPLAY WITH SPACING)

json_files = [
    'DATA/CLEAN/JSON/donnees_traitees_qs.json', 
    'DATA/CLEAN/JSON/donnees_traitees_the.json',
    'DATA/CLEAN/JSON/donnees_traitees_the_2012.json', 
    'DATA/CLEAN/JSON/donnees_traitees_the_2021.json'
]

tokens_by_continent = {}

print("\n=== LOADING REGIONAL DATA ===")
for json_path in json_files:
    try:
        if not os.path.exists(json_path): continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            docs_tokens = data.get('tokens', {})
            mapping_regions = data.get('regions', {}) 
            
            if not mapping_regions: continue

            for uni_name, tokens in docs_tokens.items():
                region = mapping_regions.get(uni_name, "Unknown")
                if region and str(region).lower() != 'nan' and region != "Unknown":
                    if region not in tokens_by_continent:
                        tokens_by_continent[region] = []
                    tokens_by_continent[region].extend(tokens)
    except Exception as e:
        print(f"Error: {e}")

# --- PLOT 1 : REGIONAL OVERVIEW ---
print("\n=== PLOT 1 : REGIONAL OVERVIEW (TOP 8 REGIONS) ===")

valid_regions = {r: t for r, t in tokens_by_continent.items() if len(t) > 1000}
sorted_regions = sorted(valid_regions.keys(), key=lambda r: len(valid_regions[r]), reverse=True)
top_regions = sorted_regions[:8]

def plot_region_batch(region_list, batch_name):
    """Generates a clean 2x2 grid with Extra Vertical Spacing"""
    if not region_list: return

    nb_regions = len(region_list)
    cols = 2 
    rows = math.ceil(nb_regions / cols)

    # Increased height (6 -> 7 per row) to help spacing
    fig, axes = plt.subplots(rows, cols, figsize=(16, 7 * rows)) 
    
    if nb_regions > 1: axes_flat = axes.flatten()
    else: axes_flat = [axes]

    print(f"\n--- Generating {batch_name} ---")

    for i, region in enumerate(region_list):
        ax = axes_flat[i]
        tokens = tokens_by_continent[region]
        text = " ".join(tokens)
        
        wc = WordCloud(width=800, height=500, background_color='white', collocations=False, 
                       max_words=20, min_font_size=15, colormap='Dark2').generate(text)
        
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f"{region.upper()}\n({len(tokens)} words)", fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_edgecolor('#cccccc'); spine.set_linewidth(1)

    # Hide empty axes
    for j in range(nb_regions, len(axes_flat)):
        axes_flat[j].axis('off')
        axes_flat[j].set_visible(False)

    plt.suptitle(f"Regional Overview - {batch_name}", fontsize=22, fontweight='bold', y=0.98)
    
    # --- CRITICAL FIX: FORCE VERTICAL SPACE (HSPACE) ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4) # 0.4 adds significant vertical gap
    # ---------------------------------------------------
    
    plt.show()

if not top_regions:
    print("No regional data found.")
else:
    batch_size = 4
    for i in range(0, len(top_regions), batch_size):
        batch = top_regions[i : i + batch_size]
        plot_region_batch(batch, f"PART {i//batch_size + 1}")


# --- PLOT 2 : ZOOM STRATÉGIQUE ---
print("\n=== PLOT 2 : STRATEGIC ZOOM (NA, EUROPE, ASIA) ===")

target_regions = ["North America", "Europe", "Asia"]
TOP_N_ZOOM = 30 
regions_subset = [r for r in target_regions if r in tokens_by_continent]

if regions_subset:
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 8))
    if len(regions_subset) == 1: axes2 = [axes2]
    
    for i, region in enumerate(regions_subset):
        ax = axes2[i]
        tokens = tokens_by_continent[region]
        print_top_words(tokens, region.upper(), TOP_N_ZOOM)

        text = " ".join(tokens)
        wc = WordCloud(width=800, height=500, background_color='white', collocations=False, 
                       max_words=TOP_N_ZOOM, min_font_size=12, colormap='tab10').generate(text)
        
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f"--- {region.upper()} ---", fontsize=20, fontweight='bold', color='darkblue', pad=20)
        ax.axis('off')
        
    for j in range(len(regions_subset), 3):
        axes2[j].axis('off')
        axes2[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
# 3 - Word comparison between THE and QS files

'''

# File paths
# Ensure CSV is the mapping file (correspondence between QS and THE names)
file_matches = 'DATA/CLEAN/CSV/university_mapping_qs_the.csv'

# New paths to JSONs
path_json_qs = 'DATA/CLEAN/JSON/donnees_traitees_qs.json'
path_json_the = 'DATA/CLEAN/JSON/donnees_traitees_the.json'

# Matching threshold (Name similarity score)
SCORE_THRESHOLD = 0.857

# CSV Column names
COL_QS_NAME = 'QS_Name'   
COL_THE_NAME = 'THE_Name'
COL_SCORE = 'Score'

# ==============================================================================
# 1. DATA LOADING
# ==============================================================================
print("=== 1. LOADING ===")

# A. Load and filter CSV (The "Judge")
try:
    df_matches = pd.read_csv(file_matches)
    # Keep only lines with good match score
    df_filtered = df_matches[df_matches[COL_SCORE] > SCORE_THRESHOLD]
    print(f"-> CSV loaded. Valid pairs (> {SCORE_THRESHOLD}) : {len(df_filtered)}")
except FileNotFoundError:
    print(f"Error : Cannot find CSV file {file_matches}")
    exit()

# B. JSON loading function
def load_json_tokens(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # IMPORTANT : In your JSON, words are under the "tokens" key
            # Structure : { "info": ..., "tokens": { "Univ": ["word1"] }, ... }
            return data.get('tokens', {}) 
    except FileNotFoundError:
        print(f"Error : Cannot find {path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error : File {path} is not valid JSON.")
        return {}

print("-> Loading QS JSON...")
docs_qs = load_json_tokens(path_json_qs)

print("-> Loading THE JSON...")
docs_the = load_json_tokens(path_json_the)


# ==============================================================================
# 2. CROSS-REFERENCING AND FILTERING
# ==============================================================================
print("\n=== 2. DATA CROSS-REFERENCING ===")

tokens_qs_final = []
tokens_the_final = []
count_match = 0
missing_qs = 0
missing_the = 0

# Iterate through filtered CSV
for index, row in df_filtered.iterrows():
    name_qs = str(row[COL_QS_NAME]).strip()
    name_the = str(row[COL_THE_NAME]).strip()
    
    # Check if university exists in our JSON files
    # (Use .get() to avoid errors if key doesn't exist)
    words_qs = docs_qs.get(name_qs)
    words_the = docs_the.get(name_the)

    if words_qs and words_the:
        # If university found in BOTH JSON files
        tokens_qs_final.extend(words_qs)
        tokens_the_final.extend(words_the)
        count_match += 1
    else:
        # Just for debug, see why it doesn't match
        if not words_qs: missing_qs += 1
        if not words_the: missing_the += 1

print(f"-> Analysis based on {count_match} common universities (present in CSV + JSONs).")
if missing_qs > 0 or missing_the > 0:
    print(f"-> Warning : {missing_qs} QS universities and {missing_the} THE universities from CSV not found in JSONs (exact name issue?).")

print(f"-> Total QS words  : {len(tokens_qs_final)}")
print(f"-> Total THE words : {len(tokens_the_final)}")


# ==============================================================================
# 3. VISUALIZATION
# ==============================================================================
print("\n=== 3. GENERATING WORD CLOUDS ===")

def plot_clouds(tokens1, tokens2):
    # List -> Text transformation
    text1 = " ".join(tokens1)
    text2 = " ".join(tokens2)
    
    if not text1 or not text2:
        print("Error : Not enough words to generate clouds.")
        return

    # Creating WordClouds
    # Limit to 50 words for readability
    wc_qs = WordCloud(width=800, height=400, background_color='white', 
                      colormap='Blues', collocations=False, max_words=50).generate(text1)
                      
    wc_the = WordCloud(width=800, height=400, background_color='white', 
                       colormap='Reds', collocations=False, max_words=50).generate(text2)

    # Display
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    axes[0].imshow(wc_qs, interpolation='bilinear')
    axes[0].set_title(f"QS 2025 Vocabulary\n(On {count_match} common universities)", fontsize=16, color='darkblue', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(wc_the, interpolation='bilinear')
    axes[1].set_title(f"THE 2025 Vocabulary\n(On {count_match} common universities)", fontsize=16, color='darkred', fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Run
if count_match > 0:
    plot_clouds(tokens_qs_final, tokens_the_final)
else:
    print("No matches found. Check that CSV names are EXACTLY the same as JSON keys.")

'''
# =============================================================================
# 4 - Combined Co-occurrence Graph (QS + THE)

# CONFIGURATION

# List of JSON files to combine (Only QS and THE as requested)
files_to_combine = [
    'DATA/CLEAN/JSON/donnees_traitees_qs.json',
    'DATA/CLEAN/JSON/donnees_traitees_the.json',
]

# --- GRAPH PARAMETERS ---
# Number of most frequent words to display.
TOP_N_WORDS = 20

# Minimum co-occurrence threshold
# An edge is drawn only if both words appear together in X documents
MIN_EDGE_WEIGHT = 10 

# LOADING AND AGGREGATION

print("=== 1. LOADING AND MERGING CORPORA (QS + THE) ===")

all_docs_list = [] # List containing word lists of ALL universities

for file_path in files_to_combine:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 1. Load JSON
            data = json.load(f)
            
            # 2. Get tokens
            docs_tokens = data.get('tokens', {})
            
            # 3. Add to global list
            count_local = 0
            for tokens in docs_tokens.values():
                if tokens: # Avoid empty lists
                    all_docs_list.append(tokens)
                    count_local += 1
                
            print(f"-> Successfully loaded : {os.path.basename(file_path)} ({count_local} universities)")
        
    except FileNotFoundError:
        print(f" /!\\ Error : File not found -> {file_path}")
    except json.JSONDecodeError:
        print(f" /!\\ Error : Corrupted JSON -> {file_path}")
    except Exception as e:
        print(f" /!\\ Error on {file_path} : {e}")

print(f"\n-> TOTAL DOCUMENTS ANALYZED : {len(all_docs_list)}")

# STATISTICAL CALCULATIONS
print("\n=== 2. FREQUENCY AND CO-OCCURRENCE ANALYSIS ===")

if len(all_docs_list) == 0:
    print("Error : No documents loaded. Check your file paths.")
    exit()

# A. Selecting Top Words on the combined set
all_tokens_flat = [token for doc in all_docs_list for token in doc]
word_counts = Counter(all_tokens_flat)

# Get top N most frequent words
top_words_dict = dict(word_counts.most_common(TOP_N_WORDS))
top_words_set = set(top_words_dict.keys())

print(f"-> Top {TOP_N_WORDS} words selected (ex: {list(top_words_dict.keys())[:5]}...)")

# B. Calculating Co-occurrences
co_occurrence_counts = Counter()

for tokens in all_docs_list:
    # 1. Keep only Top N words present in this document
    filtered_tokens = [t for t in tokens if t in top_words_set]
    
    # 2. Unique words per document
    unique_tokens = sorted(list(set(filtered_tokens)))
    
    # 3. Pairs
    if len(unique_tokens) > 1:
        pairs = list(combinations(unique_tokens, 2))
        co_occurrence_counts.update(pairs)

print(f"-> Links calculated. Total unique pairs found : {len(co_occurrence_counts)}")

# GRAPH CONSTRUCTION (NetworkX)
print("\n=== 3. GENERATING COMBINED GRAPH ===")

G = nx.Graph()

# A. Adding Nodes
for word, count in top_words_dict.items():
    G.add_node(word, size=count)

# B. Adding Edges
edges_added = 0
for pair, weight in co_occurrence_counts.items():
    if weight >= MIN_EDGE_WEIGHT:
        G.add_edge(pair[0], pair[1], weight=weight)
        edges_added += 1

print(f"-> Final graph : {G.number_of_nodes()} nodes, {edges_added} links.")

# --- NEW SECTION: PRINT NODES AND LINKS DETAILS ---
print("\n" + "="*60)
print(">>> GRAPH DATA DETAILS (NODES & EDGES) <<<")
print("="*60)

# 1. Print Nodes
print(f"\n--- NODES (Words sorted by Frequency) ---")
print(f"{'Rank':<5} | {'Word':<20} | {'Size (Freq)':<10}")
print("-" * 45)
# Sort nodes by size attribute
sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[1]['size'], reverse=True)
for i, (node, data) in enumerate(sorted_nodes, 1):
    print(f"{i:<5} | {node:<20} | {data['size']:<10}")

# 2. Print Edges
print(f"\n--- EDGES (Links sorted by Weight/Co-occurrence) ---")
print(f"{'Word A':<20} <--> {'Word B':<20} | {'Weight':<10}")
print("-" * 60)
# Sort edges by weight attribute
sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
for u, v, data in sorted_edges:
    print(f"{u:<20} <--> {v:<20} | {data['weight']:<10}")
print("="*60 + "\n")
# --------------------------------------------------

# VISUALIZATION

plt.figure(figsize=(18, 14))

# 1. Layout
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# 2. Sizes
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

# 3. Drawing
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#69b3a2', alpha=0.9, edgecolors='white')

# Labels (INCREASED SIZE HERE)
nx.draw_networkx_labels(G, pos, font_size=14, font_family='sans-serif', font_weight='bold')

nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, edge_color='gray')

plt.title(f"Co-occurrence Network (QS & THE) - Top {TOP_N_WORDS} words", fontsize=20)
plt.axis('off')

plt.figtext(0.5, 0.02, 
            f"Based on {len(all_docs_list)} descriptions.\n"
            f"Link shown if co-occurrence >= {MIN_EDGE_WEIGHT}.", 
            ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

plt.show()
# -------------------------------------------------------------------------------
# 5 - Temporal frequency analysis 

# --- CONFIGURATION ---
files_timeline = [
    {'year': '2012', 'source': 'THE', 'path': 'DATA/CLEAN/JSON/donnees_traitees_the_2012.json'},
    {'year': '2021', 'source': 'THE', 'path': 'DATA/CLEAN/JSON/donnees_traitees_the_2021.json'},
    {'year': '2025', 'source': 'THE', 'path': 'DATA/CLEAN/JSON/donnees_traitees_the.json'}
]

# MAINTAIN UNIVERSITY BALANCE
TOP_N_UNIV_LIMIT = 200 

# Words to analyze
KEYWORDS_OLD = ["founded", "science", "teach"]
KEYWORDS_NEW = ["sustainable", "impact", "global", "collaboration", "innovation", "cultural", "people"]
KEYWORDS = KEYWORDS_OLD + KEYWORDS_NEW

# --- NEW COLOR PALETTE ---
COLOR_MAP = {
    # OLD WORLD (Warm Tones / Earth / Past)
    "founded": "#d62728",      # Brick red
    "science": "#ff7f0e",      # Orange 
    "teach": "#8c564b",        # Earth brown

    # NEW WORLD (Cool Tones / Bright / Future)
    "sustainable": "#2ca02c",  # Green (Ecology)
    "impact": "#1f77b4",       # Standard blue (Action)
    "global": "#17becf",       # Cyan (International)
    "collaboration": "#9467bd",# Purple (Connection)
    "innovation": "#e377c2",   # Fuchsia pink (Modernity)
    "cultural": "#bcbd22",     # Olive yellow (Diversity)
    "people": "#000080"        # Navy blue (Human)
}

results = []
print("=== VOLUMETRIC ANALYSIS (FULL TEXT) ===")

for item in files_timeline:
    try:
        with open(item['path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 1. Select the Elite (Top 200)
            docs_tokens = dict(list(data.get('tokens', {}).items())[:TOP_N_UNIV_LIMIT])
            
            # 2. Take ALL words
            all_tokens = []
            for tokens in docs_tokens.values():
                all_tokens.extend(tokens)
            
            # 3. Normalization (Base 10,000 words)
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
        print(f"Error : {e}")

# --- VISUALIZATION ---
df = pd.DataFrame(results)

if not df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    sns.set_style("whitegrid")

    # Graph 1: Anchoring (Old World)
    for word in KEYWORDS_OLD:
        # Use .get() with default black color just in case
        color = COLOR_MAP.get(word, '#000000')
        sns.lineplot(data=df, x='Year_Label', y=word, color=color, 
                     marker='o', linestyle='--', linewidth=3, ax=axes[0], label=word.upper())
    
    axes[0].set_title("Historical & Academic Anchoring (Declining)", fontsize=14, fontweight='bold', color='#8c564b')
    axes[0].set_ylabel("Occurrences per 10,000 words")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Graph 2: Impact (New World)
    for word in KEYWORDS_NEW:
        color = COLOR_MAP.get(word, '#000000')
        sns.lineplot(data=df, x='Year_Label', y=word, color=color, 
                     marker='s', linestyle='-', linewidth=3, ax=axes[1], label=word.upper())
    
    axes[1].set_title("Societal & Human Impact (Rising)", fontsize=14, fontweight='bold', color='#1f77b4')
    axes[1].set_ylabel("Occurrences per 10,000 words")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("The Great Semantic Shift (2012-2025)", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()
    
    # Display raw data
    print("\n--- CALCULATED DATA (Freq. / 10k words) ---")
    print(df.set_index('Year_Label')[KEYWORDS].round(1))
else:
    print("No data.")