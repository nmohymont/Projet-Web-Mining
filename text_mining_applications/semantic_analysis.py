import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

files_config = [
    {'path': 'DATA/CLEAN/JSON/university_processed_features_the.json', 'source': 'THE 2025'},
    {'path': 'DATA/CLEAN/JSON/university_processed_features_qs.json',  'source': 'QS 2025'}
]

# Thésaurus (Votre grille de lecture)
thesaurus = {
    'EXCELLENCE': ['leading', 'world', 'top', 'rank', 'excellence', 'best', 'prestigous', 'elite', 'reputation', 'award', 'nobel', 'founded', 'history'],
    'INNOVATION': ['innovation', 'technology', 'digital', 'future', 'modern', 'new', 'creative', 'tech', 'smart', 'entrepreneur', 'solution'],
    'SOCIETAL': ['community', 'social', 'public', 'sustainable', 'environment', 'health', 'diversity', 'inclusion', 'global', 'impact', 'sdg'],
    'CARRIERE': ['career', 'job', 'employability', 'industry', 'business', 'skill', 'professional', 'work', 'alumni', 'salary'],
    'RECHERCHE': ['research', 'study', 'science', 'knowledge', 'academic', 'theory', 'publication', 'institute', 'phd', 'faculty', 'professor']
}

# ==============================================================================
# 2. MOTEUR D'ANALYSE (DÉTAILLÉ)
# ==============================================================================

def analyser_universites_detail(files_list, thesaurus_dict):
    results = []
    word_details = [] 
    
    print("=== ANALYSE SÉMANTIQUE DÉTAILLÉE ===")

    for file_info in files_list:
        filepath = file_info['path']
        source = file_info['source']
        
        if not os.path.exists(filepath):
            print(f"/!\\ Fichier introuvable : {filepath} (Ignoré)")
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tokens_dict = data.get('tokens', {})
                regions_dict = data.get('regions', {})
                
                print(f"-> {source} : {len(tokens_dict)} universités analysées.")
                
                for uni_name, tokens in tokens_dict.items():
                    region = regions_dict.get(uni_name, "Autre")
                    scores = {cat: 0 for cat in thesaurus_dict.keys()}
                    local_word_counts = {}
                    
                    if tokens:
                        for token in tokens:
                            token_lower = token.lower()
                            for cat, keywords in thesaurus_dict.items():
                                if token_lower in keywords:
                                    scores[cat] += 1
                                    local_word_counts[token_lower] = local_word_counts.get(token_lower, 0) + 1
                    
                    total_words = len(tokens)
                    max_score = max(scores.values()) if scores else 0
                    dominant_cat = max(scores, key=scores.get) if max_score > 0 else "Neutre"
                    
                    results.append({
                        'Université': uni_name,
                        'Source': source,
                        'Région': region,
                        'Sentiment Dominant': dominant_cat,
                        'Total Words': total_words,
                        **scores
                    })

                    for word, count in local_word_counts.items():
                        found_cat = "Inconnu"
                        for cat, kws in thesaurus_dict.items():
                            if word in kws:
                                found_cat = cat
                                break
                        word_details.append({
                            'Source': source,
                            'Category': found_cat,
                            'Word': word,
                            'Count': count
                        })
                    
        except Exception as e:
            print(f"Erreur avec {filepath}: {e}")
            
    return pd.DataFrame(results), pd.DataFrame(word_details)

# ==============================================================================
# 3. FONCTIONS GRAPHIQUES ET AFFICHAGE
# ==============================================================================

def plot_bar_global(df):
    """ GRAPHIQUE 1 : Global (THE vs QS) """
    
    # --- PRINT DATA FOR PROMPT ---
    print("\n>>> DATA: GLOBAL DISTRIBUTION (THE vs QS) <<<")
    print("-" * 50)
    dist_data = df.groupby(['Source', 'Sentiment Dominant']).size().unstack(fill_value=0)
    print(dist_data)
    print("-" * 50)
    # -----------------------------

    plt.figure(figsize=(10, 6))
    order = df['Sentiment Dominant'].value_counts().index
    sns.countplot(data=df, x='Sentiment Dominant', hue='Source', order=order, palette='viridis')
    plt.title("Répartition Globale des Profils (THE vs QS)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

def plot_radar(df, categories):
    """ GRAPHIQUE 2 : Radar (Profil Moyen) """
    df_radar = df.groupby('Source')[categories + ['Total Words']].sum()
    for cat in categories:
        df_radar[cat] = (df_radar[cat] / df_radar['Total Words']) * 1000
    
    # --- PRINT DATA FOR PROMPT ---
    print("\n" + "="*60)
    print(">>> DATA: RADAR SCORES (Points per 1000 words) <<<")
    print("="*60)
    print(df_radar[categories].round(2))
    print("="*60 + "\n")
    # -----------------------------

    df_radar = df_radar.drop(columns=['Total Words'])
    
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, size=10, fontweight='bold')
    plt.ylim(0, df_radar.max().max() * 1.1)
    
    colors = ['#1f77b4', '#ff7f0e']
    for i, (src, row) in enumerate(df_radar.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, linewidth=2, label=src, color=colors[i%2])
        ax.fill(angles, values, color=colors[i%2], alpha=0.1)
    
    plt.title("Profil Sémantique Moyen (THE vs QS)", size=14, y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.show()

def plot_region_trend(df):
    """ GRAPHIQUE 3 : Régions """
    target_regions = ['North America', 'Europe', 'Asia']
    df_reg = df[df['Région'].isin(target_regions)].copy()
    
    if df_reg.empty:
        return

    df_prop = df_reg.groupby(['Région', 'Sentiment Dominant']).size().reset_index(name='Count')
    region_totals = df_prop.groupby('Région')['Count'].transform('sum')
    df_prop['Pourcentage'] = (df_prop['Count'] / region_totals) * 100

    # --- PRINT DATA FOR PROMPT ---
    print("\n>>> DATA: REGIONAL TRENDS (Percentage) <<<")
    print("-" * 60)
    pivot_reg = df_prop.pivot(index='Région', columns='Sentiment Dominant', values='Pourcentage').fillna(0).round(1)
    print(pivot_reg)
    print("-" * 60)
    # -----------------------------

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.barplot(data=df_prop, x='Région', y='Pourcentage', hue='Sentiment Dominant', palette='rocket')
    plt.title("Comparaison Culturelle par Continent", fontsize=14, fontweight='bold')
    plt.ylabel("Part des Universités (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_thesaurus_breakdown(df_details, thesaurus_dict):
    """ GRAPHIQUE 4 : Détail par mot (FUSIONNÉ EN UNE SEULE FIGURE) """
    print("\n=== DÉTAIL DU THÉSAURUS (CONTRIBUTION PAR MOT) ===")
    
    # Agrégation globale
    df_agg = df_details.groupby(['Source', 'Category', 'Word'])['Count'].sum().reset_index()
    
    # Calcul pourcentage de contribution
    cat_totals = df_agg.groupby(['Source', 'Category'])['Count'].transform('sum')
    df_agg['Contribution (%)'] = (df_agg['Count'] / cat_totals) * 100
    
    # --- PRINT TOP WORDS FOR PROMPT ---
    print("\n>>> DATA: TOP CONTRIBUTING WORDS PER CATEGORY <<<")
    for cat in thesaurus_dict.keys():
        print(f"\n--- {cat} ---")
        top_words = df_agg[df_agg['Category'] == cat].groupby('Word')['Count'].sum().sort_values(ascending=False).head(5)
        print(top_words)
    print("="*60)
    # ----------------------------------

    sources = df_agg['Source'].unique()
    categories = list(thesaurus_dict.keys())
    
    n_rows = len(sources)
    n_cols = len(categories)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), sharey=True, constrained_layout=True, squeeze=False)

    for i, source in enumerate(sources):
        for j, cat in enumerate(categories):
            ax = axes[i][j]
            
            subset = df_agg[(df_agg['Source'] == source) & (df_agg['Category'] == cat)]
            subset = subset.sort_values(by='Count', ascending=False).head(8)
            
            if not subset.empty:
                sns.barplot(data=subset, x='Contribution (%)', y='Word', ax=ax, palette='Blues_r')
                
                if i == 0: 
                    ax.set_title(cat, fontsize=12, fontweight='bold')
                
                ax.set_xlabel("%" if i == n_rows - 1 else "") 
                ax.set_ylabel(source if j == 0 else "") 
                if j == 0:
                    ax.set_ylabel(source, fontsize=12, fontweight='bold', rotation=90)

            else:
                ax.set_visible(False)
    
    plt.suptitle("De quoi sont faits les scores ? (Top Mots Contributeurs)", fontsize=16, fontweight='bold', y=1.02)
    plt.show()

# ==============================================================================
# 4. EXÉCUTION
# ==============================================================================

df_final, df_words_detail = analyser_universites_detail(files_config, thesaurus)

if not df_final.empty:
    print("\n[1/4] Graphique 1 : Répartition Globale...")
    plot_bar_global(df_final)
    
    print("\n[2/4] Graphique 2 : Radar Chart...")
    plot_radar(df_final, list(thesaurus.keys()))
    
    print("\n[3/4] Graphique 3 : Tendance Régionale...")
    plot_region_trend(df_final)
    
    print("\n[4/4] Graphique 4 : Détail par Mot (Fusionné)...")
    if not df_words_detail.empty:
        plot_thesaurus_breakdown(df_words_detail, thesaurus)
else:
    print("Erreur : Aucune donnée chargée.")