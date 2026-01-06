import json
import os
from collections import Counter

# ==============================================================================
# CONFIGURATION
# ==============================================================================
file_path = 'DATA/CLEAN/JSON/university_processed_features_qs.json'

# List of your 6 target words (preferably in lowercase)
#'''
target_words = [
    "intelligence",
    "artificial",
]
#'''
'''
target_words = [
    "prime",
    "minister",
]
'''
'''
target_words = [
    "computer",
    "electrical",
    "civil",
    "mechanical",
    "chemical",
    "engineering"
]
'''
'''
target_words = [
    "dentistry",
    "nursing",
    "pharmacy"
]
'''
'''
target_words = [
    "facebook",
    "youtube",
    "instagram"
]
'''
'''
target_words = [
    "finance",
    "accounting"
]
'''
'''
target_words = [
    "environmental",
    "sustainability"
]
'''
'''
target_words = [
    "highly",
    "qualified"
]
'''

# ==============================================================================
# 1. LOADING DATA
# ==============================================================================
if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found.")
    if os.path.exists('university_processed_features_qs.json'):
        file_path = 'university_processed_features_qs.json'
        print(f"File found locally: {file_path}")
    else:
        exit()

with open(file_path, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

tokens_dict = full_data.get('tokens', {})

# Attempting to retrieve regions/continents
# We look for 'region' or 'regions' in the JSON
regions_dict = full_data.get('region', full_data.get('regions', {}))

# ==============================================================================
# 2. SEMANTIC ANALYSIS
# ==============================================================================
results = []
print(f"Processing {len(tokens_dict)} universities for {len(target_words)} target words...\n")

for uni_name, tokens in tokens_dict.items():
    tokens_set = set(t.lower() for t in tokens)
    
    found_words = [word for word in target_words if word in tokens_set]
    missing_words = [word for word in target_words if word not in tokens_set]
    
    match_count = len(found_words)
    
    # We keep the university if it has at least 1 word
    if match_count > 0:
        # We retrieve the region here (or "Unknown" if not in JSON)
        uni_region = regions_dict.get(uni_name, "Region not specified")
        
        results.append({
            "name": uni_name,
            "count": match_count,
            "found": found_words,
            "region": uni_region # Storing the region
        })

# Sort: number of words (descending), then name
results.sort(key=lambda x: (-x['count'], x['name']))

perfect_matches = [r for r in results if r['count'] == len(target_words)]
partial_matches = [r for r in results if 0 < r['count'] < len(target_words)]

# ==============================================================================
# 3. DISPLAY RESULTS
# ==============================================================================
print("-" * 80)
print(f"ANALYSIS REPORT FOR: {', '.join(target_words).upper()}")
print("-" * 80)

print(f"Total universities analyzed: {len(tokens_dict)}")
print(f"Universities with ALL words ({len(target_words)}/{len(target_words)}): {len(perfect_matches)}")
print(f"Universities with partial results: {len(partial_matches)}")

print("\n" + "="*80)
print(f"TOP RESULTS: PERFECT MATCH")
print("="*80)

if perfect_matches:
    for i, res in enumerate(perfect_matches, 1):
        # We also display the region next to the name
        print(f"{i}. {res['name']} ({res['region']})")
else:
    print("No perfect match found.")

print("\n" + "="*80)
print(f"PARTIAL RESULTS (Top 20)")
print("="*80)

limit_display = 20 
if partial_matches:
    for i, res in enumerate(partial_matches[:limit_display], 1):
        found_str = ", ".join(res['found'])
        print(f"{i}. [{res['count']}/{len(target_words)}] {res['name']} ({res['region']})")
        print(f"   -> Found: {found_str}")
else:
    print("No partial results found.")


# ==============================================================================
# 4. STATISTICS PER WORD
# ==============================================================================
print("\n" + "="*80)
print("OCCURRENCE STATISTICS PER WORD")
print("="*80)
for word in target_words:
    count = sum(1 for tokens in tokens_dict.values() if word in [t.lower() for t in tokens])
    print(f"- '{word}' appears in {count} universities.")


# ==============================================================================
# 5. GEOGRAPHIC ANALYSIS (PERFECT MATCHES ONLY)
# ==============================================================================
print("\n" + "="*80)
print(f"GEOGRAPHIC DISTRIBUTION FOR PERFECT MATCHES ONLY")
print(f"(Regions where universities have ALL {len(target_words)} words)")
print("="*80)

if not regions_dict:
    print("⚠️ Warning: No region data found in the JSON file.")
elif not perfect_matches:
    print("No perfect matches found, so no regional breakdown possible.")
else:
    # We only extract regions from the 'perfect_matches' list
    perfect_regions = [res['region'] for res in perfect_matches]
    
    # Count occurrences
    region_stats = Counter(perfect_regions)
    
    # Display sorted by descending count
    for region, count in region_stats.most_common():
        percentage = (count / len(perfect_matches)) * 100
        print(f"- {region:<25} : {count:>3} universities ({percentage:.1f}%)")