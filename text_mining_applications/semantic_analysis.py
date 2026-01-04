import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# ==============================================================================
# CONFIGURATION & THESAURUS
# ==============================================================================

files_config = [
    {'path': 'DATA/CLEAN/JSON/university_processed_features_the.json', 'source': 'THE 2025'},
    {'path': 'DATA/CLEAN/JSON/university_processed_features_qs.json',  'source': 'QS 2025'}
]

# Le prisme de lecture
thesaurus = {
    'EXCELLENCE': ['leading', 'world', 'top', 'rank', 'excellence', 'best', 'prestigous', 'elite', 'reputation', 'award', 'nobel', 'founded', 'history'],
    'INNOVATION': ['innovation', 'technology', 'digital', 'future', 'modern', 'new', 'creative', 'tech', 'smart', 'entrepreneur', 'solution'],
    'SOCIETAL': ['community', 'social', 'public', 'sustainable', 'environment', 'health', 'diversity', 'inclusion', 'global', 'impact', 'sdg'],
    'CARRIERE': ['career', 'job', 'employability', 'industry', 'business', 'skill', 'professional', 'work', 'alumni', 'salary'],
    'RECHERCHE': ['research', 'study', 'science', 'knowledge', 'academic', 'theory', 'publication', 'institute', 'phd', 'faculty', 'professor']
}

# ==============================================================================
# MOTEUR D'ANALYSE
# ==============================================================================

def load_and_analyze(files_list, thesaurus_dict):
    results = []
    word_details = [] 
    
    print("=== CHARGEMENT ET ANALYSE DES DONNÉES ===")

    for file_info in files_list:
        filepath = file_info['path']
        source = file_info['source']
        
        if not os.path.exists(filepath):
            print(f"/!\\ Fichier introuvable : {filepath}")
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
                        found_cat = next((cat for cat, kws in thesaurus_dict.items() if word in kws), "Inconnu")
                        word_details.append({'Source': source, 'Category': found_cat, 'Word': word, 'Count': count})
                    
        except Exception as e:
            print(f"Erreur avec {filepath}: {e}")
            
    return pd.DataFrame(results), pd.DataFrame(word_details)

# --- EXÉCUTION DU CHARGEMENT ---
df_final, df_words_detail = load_and_analyze(files_config, thesaurus)
print("Done. Données prêtes dans 'df_final'.")


# ==============================================================================
# PLOT 1 : GLOBAL DISTRIBUTION (THE vs QS)
# ==============================================================================

if not df_final.empty:
    print("\n>>> DATA: GLOBAL DISTRIBUTION (THE vs QS) <<<")
    print("-" * 50)
    print(df_final.groupby(['Source', 'Sentiment Dominant']).size().unstack(fill_value=0))
    print("-" * 50)

    plt.figure(figsize=(10, 6))
    
    # Ordre décroissant pour plus de lisibilité
    order = df_final['Sentiment Dominant'].value_counts().index
    
    sns.countplot(
        data=df_final, 
        x='Sentiment Dominant', 
        hue='Source', 
        order=order, 
        palette='viridis'
    )
    
    plt.title("Répartition Globale des Profils (THE vs QS)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

# ==============================================================================
# PLOT 2 : RADAR CHART (INTENSITY PER 1000 WORDS)
# ==============================================================================

categories = list(thesaurus.keys())

if not df_final.empty:
    # 1. Calcul des moyennes pondérées par 1000 mots
    df_radar = df_final.groupby('Source')[categories + ['Total Words']].sum()
    for cat in categories:
        df_radar[cat] = (df_radar[cat] / df_radar['Total Words']) * 1000
    
    print("\n>>> DATA: RADAR SCORES (Points per 1000 words) <<<")
    print(df_radar[categories].round(2))

    df_radar = df_radar.drop(columns=['Total Words'])
    
    # 2. Préparation géométrique
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Fermer la boucle
    
    # 3. Tracé
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, size=10, fontweight='bold')
    plt.ylim(0, df_radar.max().max() * 1.1)
    
    colors = ['#1f77b4', '#ff7f0e'] # Bleu (THE) / Orange (QS)
    
    for i, (src, row) in enumerate(df_radar.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, linewidth=2, label=src, color=colors[i%2])
        ax.fill(angles, values, color=colors[i%2], alpha=0.1)
    
    plt.title("Profil Sémantique Moyen (Intensité)", size=14, y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.show()

# ==============================================================================
# PLOT 3 : REGIONAL TRENDS (Optimized Colors & Filtered)
# ==============================================================================

target_regions = ['North America', 'Europe', 'Asia']
# 1. Filtre sur les régions
df_reg = df_final[df_final['Région'].isin(target_regions)].copy()

# 2. Filtre pour EXCLURE 'CARRIERE' et 'Neutre'
# Le tilde (~) signifie "inverse de" -> On garde tout ce qui N'EST PAS dans la liste
exclude_categories = ['CARRIERE', 'Neutre']
df_reg = df_reg[~df_reg['Sentiment Dominant'].isin(exclude_categories)]

if not df_reg.empty:
    # 3. Calcul des pourcentages (Renormalisé sur les catégories restantes)
    df_prop = df_reg.groupby(['Région', 'Sentiment Dominant']).size().reset_index(name='Count')
    region_totals = df_prop.groupby('Région')['Count'].transform('sum')
    df_prop['Pourcentage'] = (df_prop['Count'] / region_totals) * 100

    print("\n>>> DATA: REGIONAL TRENDS (Percentage - Filtered) <<<")
    print(df_prop.pivot(index='Région', columns='Sentiment Dominant', values='Pourcentage').fillna(0).round(1))

    # 4. Tracé
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    sns.barplot(
        data=df_prop, 
        x='Région', 
        y='Pourcentage', 
        hue='Sentiment Dominant', 
        # Changement de palette pour 'tab10' (couleurs très distinctes)
        palette='tab10' 
    )
    
    plt.title("Comparaison Culturelle par Continent (Sans Carrière/Neutre)", fontsize=14, fontweight='bold')
    plt.ylabel("Part des Universités (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Dominante")
    plt.tight_layout()
    plt.show()
else:
    print("Aucune donnée disponible après filtrage.")
# ==============================================================================
# PLOT 4 : TOP CONTRIBUTING WORDS
# ==============================================================================

if not df_words_detail.empty:
    print("\n=== DÉTAIL DU THÉSAURUS (CONTRIBUTION PAR MOT) ===")
    
    # 1. Agrégation et calcul de contribution
    df_agg = df_words_detail.groupby(['Source', 'Category', 'Word'])['Count'].sum().reset_index()
    cat_totals = df_agg.groupby(['Source', 'Category'])['Count'].transform('sum')
    df_agg['Contribution (%)'] = (df_agg['Count'] / cat_totals) * 100
    
    # 2. Setup grille
    sources = df_agg['Source'].unique()
    categories_list = list(thesaurus.keys())
    
    n_rows = len(sources)
    n_cols = len(categories_list)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), sharey=True, constrained_layout=True, squeeze=False)

    for i, source in enumerate(sources):
        for j, cat in enumerate(categories_list):
            ax = axes[i][j]
            
            # Top 8 mots
            subset = df_agg[(df_agg['Source'] == source) & (df_agg['Category'] == cat)]
            subset = subset.sort_values(by='Count', ascending=False).head(8)
            
            if not subset.empty:
                sns.barplot(data=subset, x='Contribution (%)', y='Word', ax=ax, palette='Blues_r')
                
                # Mise en forme
                if i == 0: ax.set_title(cat, fontsize=12, fontweight='bold')
                ax.set_xlabel("%" if i == n_rows - 1 else "") 
                ax.set_ylabel("") 
                if j == 0: ax.set_ylabel(source, fontsize=12, fontweight='bold', rotation=90)
            else:
                ax.set_visible(False)
    
    plt.suptitle("De quoi sont faits les scores ? (Top Mots Contributeurs)", fontsize=16, fontweight='bold', y=1.02)
    plt.show()