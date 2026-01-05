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


thesaurus = {
    'EXCELLENCE': ['leading', 'world', 'top', 'rank', 'excellence', 'best', 'prestigous', 'elite', 'reputation', 'award', 'nobel', 'founded', 'history'],
    'INNOVATION': ['innovation', 'technology', 'digital', 'future', 'modern', 'new', 'creative', 'tech', 'smart', 'entrepreneur', 'solution'],
    'SOCIETAL': ['community', 'social', 'public', 'sustainable', 'environment', 'health', 'diversity', 'inclusion', 'global', 'impact', 'sdg'],
    'CAREER': ['career', 'job', 'employability', 'industry', 'business', 'skill', 'professional', 'work', 'alumni', 'salary'],
    'RESEARCH': ['research', 'study', 'science', 'knowledge', 'academic', 'theory', 'publication', 'institute', 'phd', 'faculty', 'professor']
}

def load_and_analyze(files_list, thesaurus_dict):
    results = []
    word_details = [] 
    
    print("=== LOADING AND ANALYZING DATA ===")

    for file_info in files_list:
        filepath = file_info['path']
        source = file_info['source']
        
        if not os.path.exists(filepath):
            print(f" ERROR : File not found: {filepath}")
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tokens_dict = data.get('tokens', {})
                regions_dict = data.get('regions', {})
                
                print(f"-> {source}: {len(tokens_dict)} universities analyzed.")
                
                for uni_name, tokens in tokens_dict.items():
                    region = regions_dict.get(uni_name, "Other") # if the key uni_name is not found it returns region = 'Other'
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
                    dominant_cat = max(scores, key=scores.get) if max_score > 0 else "Neutral"
                    
                    results.append({
                        'University': uni_name,
                        'Source': source,
                        'Region': region,
                        'Dominant feeling': dominant_cat,
                        'Total Words': total_words,
                        **scores
                    })

                    for word, count in local_word_counts.items():
                        found_cat = next((cat for cat, kws in thesaurus_dict.items() if word in kws), "Unknown")
                        word_details.append({'Source': source, 'Category': found_cat, 'Word': word, 'Count': count})
                    
        except Exception as e:
            print(f"Error with {filepath}: {e}")
            
    return pd.DataFrame(results), pd.DataFrame(word_details)

# --- LOAD & RUN ---
df_final, df_words_detail = load_and_analyze(files_config, thesaurus)
print("Done. Data available in 'df_final'.")


# ==============================================================================
# PLOT 1: GLOBAL DISTRIBUTION (THE vs QS)
# ==============================================================================

if not df_final.empty:
    print("\n>>> DATA: GLOBAL DISTRIBUTION (THE vs QS) <<<")
    print("-" * 50)
    print(df_final.groupby(['Source', 'Dominant feeling']).size().unstack(fill_value=0))
    print("-" * 50)

    plt.figure(figsize=(10, 6))
    
    # Descending order for readability
    order = df_final['Dominant feeling'].value_counts().index
    
    sns.countplot(
        data=df_final, 
        x='Dominant feeling', 
        hue='Source',
        order=order, 
        palette='viridis'
    )
    
    plt.title("Global Distribution of Profiles (THE vs QS)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

# ==============================================================================
# PLOT 2: RADAR CHART (INTENSITY PER 1000 WORDS)
# ==============================================================================

categories = list(thesaurus.keys())

if not df_final.empty:
    # 1. Compute weighted averages per 1000 words
    df_radar = df_final.groupby('Source')[categories + ['Total Words']].sum()
    for cat in categories:
        df_radar[cat] = (df_radar[cat] / df_radar['Total Words']) * 1000
    
    print("\n>>> DATA: RADAR SCORES (Points per 1000 words) <<<")
    print(df_radar[categories].round(2))

    df_radar = df_radar.drop(columns=['Total Words'])
    
    # 2. Geometry preparation
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close loop
    
    # 3. Plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, size=10, fontweight='bold')
    plt.ylim(0, df_radar.max().max() * 1.1)
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue (THE) / Orange (QS)
    
    for i, (src, row) in enumerate(df_radar.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, linewidth=2, label=src, color=colors[i%2])
        ax.fill(angles, values, color=colors[i%2], alpha=0.1)
    
    plt.title("Average Semantic Profile (Intensity)", size=14, y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.show()

# ==============================================================================
# PLOT 3: REGIONAL TRENDS 
# ==============================================================================

target_regions = ['North America', 'Europe', 'Asia']

# 1. Filter on regions
df_reg = df_final[df_final['Region'].isin(target_regions)].copy()

# 2. Exclude 'CAREER' and 'Neutral'
# Tilde (~) means logical NOT -> keep what is NOT in the list
exclude_categories = ['CAREER', 'Neutral']
df_reg = df_reg[~df_reg['Dominant feeling'].isin(exclude_categories)]

if not df_reg.empty:
    # 3. Percentage calculation (renormalized over remaining categories)
    df_prop = df_reg.groupby(['Region', 'Dominant feeling']).size().reset_index(name='Count')
    region_totals = df_prop.groupby('Region')['Count'].transform('sum')
    df_prop['Percentage'] = (df_prop['Count'] / region_totals) * 100

    print("\n>>> DATA: REGIONAL TRENDS (Percentage - Filtered) <<<")
    print(df_prop.pivot(index='Region', columns='Dominant feeling', values='Percentage').fillna(0).round(1))

    # 4. Plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    sns.barplot(
        data=df_prop, 
        x='Region', 
        y='Percentage', 
        hue='Dominant feeling', 
        palette='tab10'
    )
    
    plt.title("Cultural Comparison by Continent (Without Career/Neutral)", fontsize=14, fontweight='bold')
    plt.ylabel("Share of Universities (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Dominant Profile")
    plt.tight_layout()
    plt.show()
else:
    print("No data available after filtering.")


# ==============================================================================
# PLOT 4: TOP CONTRIBUTING WORDS
# ==============================================================================
if not df_words_detail.empty:
    print("\n=== THESAURUS DETAILS (WORD CONTRIBUTION) ===")
    
    # 1. Aggregation and contribution calculation
    # Group by Source/Category/Word to count occurrences
    df_agg = df_words_detail.groupby(['Source', 'Category', 'Word'])['Count'].sum().reset_index()
    
    # Calculate total per category to get percentage
    cat_totals = df_agg.groupby(['Source', 'Category'])['Count'].transform('sum')
    df_agg['Contribution (%)'] = (df_agg['Count'] / cat_totals) * 100
    
    # Configuration
    sources = df_agg['Source'].unique()
    categories_list = list(thesaurus.keys())

    # --------------------------------------------------------------------------
    # PART 1: CONSOLE DISPLAY
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"{'>>> DATA: TOP 8 WORDS PER CATEGORY & SOURCE <<<':^70}")
    print("="*70)

    for cat in categories_list:
        print(f"\n[[ CATEGORY: {cat.upper()} ]]")
        
        for source in sources:
            # Filter data for this category and source
            subset_print = df_agg[(df_agg['Source'] == source) & (df_agg['Category'] == cat)]
            
            # Sort and take top 8
            top_print = subset_print.sort_values(by='Count', ascending=False).head(8)
            
            # Table header display
            print(f"\n   > Source: {source}")
            print(f"   {'-'*55}")
            print(f"   | {'Word':<20} | {'Count':<8} | {'Contrib (%)':<12} |")
            print(f"   {'-'*55}")
            
            # Rows display
            if top_print.empty:
                print(f"   | {'(No data)':<20} | {'-':<8} | {'-':<12} |")
            else:
                for _, row in top_print.iterrows():
                    print(f"   | {row['Word']:<20} | {row['Count']:<8} | {row['Contribution (%)']:>6.1f} %     |")
            print(f"   {'-'*55}")
        
        print("." * 70) # Separator between categories

    # --------------------------------------------------------------------------
    # PART 2: PLOT (MATPLOTLIB / SEABORN) WITHOUT TITLE
    # --------------------------------------------------------------------------
    n_rows = len(sources)
    n_cols = len(categories_list)
    
    # Figure creation
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), 
                             sharey=True, constrained_layout=True, squeeze=False)

    for i, source in enumerate(sources):
        for j, cat in enumerate(categories_list):
            ax = axes[i][j]
            
            # Data retrieval for the plot
            subset = df_agg[(df_agg['Source'] == source) & (df_agg['Category'] == cat)]
            subset = subset.sort_values(by='Count', ascending=False).head(8)
            
            if not subset.empty:
                sns.barplot(data=subset, x='Contribution (%)', y='Word', ax=ax, palette='Blues_r')
                
                # Titles and labels
                if i == 0: 
                    ax.set_title(cat, fontsize=12, fontweight='bold')
                
                ax.set_xlabel("%" if i == n_rows - 1 else "") 
                ax.set_ylabel("") 
                
                # Source label on the left
                if j == 0: 
                    ax.set_ylabel(source, fontsize=12, fontweight='bold', rotation=90)
            else:
                ax.set_visible(False)
    
    # The line plt.suptitle(...) was removed here
    plt.show()

else:
    print("No detailed data available (df_words_detail is empty).")