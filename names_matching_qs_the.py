import pandas as pd
import difflib

# Load data
df_qs = pd.read_parquet('DATA/PARQUET/qs_university_corpus.parquet')
df_the = pd.read_parquet('DATA/PARQUET/the_university_corpus.parquet')


# Extract unique names
names_qs = df_qs['name'].dropna().unique()
names_the = df_the['name'].dropna().unique()

# Pre-processing function for better matching
def normalize(text):
    if not isinstance(text, str): return ""
    text = text.lower().strip()
    text = text.replace("the ", "").replace("-", " ").replace("university of ", "").replace(" university", "")
    return text.strip()

# Create normalized dicts for quick lookup
norm_qs = {normalize(n): n for n in names_qs}
norm_the = {normalize(n): n for n in names_the}

# 1. Exact Matches (Normalized)
matches = []
matched_qs = set()
matched_the = set()

for n_qs, original_qs in norm_qs.items():
    if n_qs in norm_the:
        original_the = norm_the[n_qs]
        matches.append({
            'QS_Name': original_qs,
            'THE_Name': original_the,
            'Score': 1.0,
            'Method': 'Exact/Normalized'
        })
        matched_qs.add(original_qs)
        matched_the.add(original_the)

# 2. Fuzzy Matching for the rest
# Filter out already matched
remaining_qs = [n for n in names_qs if n not in matched_qs]
remaining_the = [n for n in names_the if n not in matched_the]

# Optimization: Use difflib.get_close_matches which is faster than calculating all pairs
# However, get_close_matches returns a list. We want the best one.
# For 1000 items, looping is fine.

print(f"Starting fuzzy matching on {len(remaining_qs)} QS universities against {len(remaining_the)} THE universities...")

# We will store results to DataFrame later
fuzzy_matches = []

for qs_name in remaining_qs:
    # Get top 1 match from remaining_the
    # cutoff=0.6 implies we want at least 60% similarity
    match = difflib.get_close_matches(qs_name, remaining_the, n=1, cutoff=0.6)
    
    if match:
        the_match = match[0]
        # Calculate actual ratio for the score column
        score = difflib.SequenceMatcher(None, qs_name, the_match).ratio()
        fuzzy_matches.append({
            'QS_Name': qs_name,
            'THE_Name': the_match,
            'Score': score,
            'Method': 'Fuzzy'
        })
    else:
        # No match found within threshold
        fuzzy_matches.append({
            'QS_Name': qs_name,
            'THE_Name': None,
            'Score': 0.0,
            'Method': 'No Match'
        })

# Combine results
all_matches = matches + fuzzy_matches
df_matches = pd.DataFrame(all_matches)

# Sort by Score desc
df_matches = df_matches.sort_values(by='Score', ascending=False)

# Save to CSV
output_filename = 'university_mapping_qs_the.csv'
df_matches.to_csv(output_filename, index=False)

# Display stats and examples
print(f"Total Matches found: {len(df_matches)}")
print(f"Exact/Normalized Matches: {len(matches)}")
print(f"Fuzzy Matches (Score > 0.8): {len(df_matches[(df_matches['Method']=='Fuzzy') & (df_matches['Score'] > 0.8)])}")

print("\n--- Top 10 High Confidence Fuzzy Matches ---")
print(df_matches[df_matches['Method']=='Fuzzy'].head(10)[['QS_Name', 'THE_Name', 'Score']])

print("\n--- Examples of Low Confidence / Potential Errors (Score 0.6 - 0.7) ---")
print(df_matches[(df_matches['Method']=='Fuzzy') & (df_matches['Score'] > 0.6) & (df_matches['Score'] < 0.7)].head(5)[['QS_Name', 'THE_Name', 'Score']])