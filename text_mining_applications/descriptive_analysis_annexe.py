import pandas as pd
import re
import os
from collections import Counter
import random

# ==============================================================================
# TEXTUAL ANALYSIS (PARQUET SOURCE) - OPTIMIZED
# ==============================================================================

# --- 1. GLOBAL CONFIGURATION ---
PARQUET_FILES = [
    'DATA/CLEAN/PARQUET/qs_university_corpus.parquet',
    'DATA/CLEAN/PARQUET/the_university_corpus.parquet'
]

# Analysis 1: The pivot word (e.g., IMPACT)
TARGET_WORD = "impact"
CONTEXT_WINDOW = 6

# Analysis 2: The trigger phrase (e.g., AIMS TO)
TRIGGER_PHRASE = "aims to"
N_GRAM_SIZE = 6

# --- 2. DATA LOADING (ONCE ONLY) ---
print("=== 1. DATA LOADING AND PREPARATION ===")

all_texts_raw = []

for f_path in PARQUET_FILES:
    print(f"-> Reading {os.path.basename(f_path)}...")
    try:
        df = pd.read_parquet(f_path)
        
        # Smart detection of the text column
        text_col = None
        candidates = ['description', 'text', 'summary', 'overview', 'full_text']
        for col in df.columns:
            if col.lower() in candidates:
                text_col = col
                break
        
        if not text_col: # Fallback to the first string column
             text_col = df.select_dtypes(include=['object']).columns[0]

        print(f"   (Column used: '{text_col}')")
        
        # Clean NaNs and convert everything to string
        texts = df[text_col].dropna().astype(str).tolist()
        all_texts_raw.extend(texts)
        
    except Exception as e:
        print(f"   /!\\ Error: {e}")

print(f"\n=> Corpus loaded: {len(all_texts_raw)} descriptions ready for analysis.")

# ==============================================================================
# ANALYSIS 1: WORD CONTEXT (E.g., What type of impact?)
# ==============================================================================
print(f"\n" + "="*60)
print(f"=== 2. TARGET WORD ANALYSIS: '{TARGET_WORD.upper()}' ===")
print("="*60)

words_before_target = []
sample_sentences = []

for text in all_texts_raw:
    # Simple tokenization (keep words, remove punctuation)
    tokens = re.findall(r'\w+', text.lower())
    
    if TARGET_WORD in tokens:
        indices = [i for i, x in enumerate(tokens) if x == TARGET_WORD]
        
        for i in indices:
            # A. Capture the adjective just before (Statistical analysis)
            if i > 0:
                prev_word = tokens[i-1]
                # Standard stopwords + common linking words
                stopwords = ["the", "a", "an", "of", "in", "to", "for", "and", "on", "with", "its", "their", "our", "have", "has"]
                if prev_word not in stopwords:
                    words_before_target.append(prev_word)
            
            # B. Capture the full sentence snippet (For example)
            start = max(0, i - CONTEXT_WINDOW)
            end = min(len(tokens), i + CONTEXT_WINDOW + 1)
            snippet = tokens[start:end]
            # Capitalize the target word
            snippet_str = " ".join([t.upper() if t == TARGET_WORD else t for t in snippet])
            sample_sentences.append(snippet_str)

# RESULTS 1
print(f"\n--- TOP 15 ADJECTIVES BEFORE '{TARGET_WORD.upper()}' ---")
if words_before_target:
    common = Counter(words_before_target).most_common(15)
    for word, freq in common:
        print(f"{word:>15}  {TARGET_WORD.upper()}  (x{freq})")
else:
    print("Not enough data.")

print(f"\n--- CONTEXT EXAMPLES ---")
if sample_sentences:
    for s in random.sample(sample_sentences, min(5, len(sample_sentences))):
        print(f"... {s} ...")

# ==============================================================================
# ANALYSIS 2: N-GRAMS (E.g., What does the university aim to do?)
# ==============================================================================
print(f"\n" + "="*60)
print(f"=== 3. SEQUENCE ANALYSIS: AFTER '{TRIGGER_PHRASE.upper()}' ===")
print("="*60)

found_sequences = []
trigger_tokens = re.findall(r'\w+', TRIGGER_PHRASE.lower())
trigger_len = len(trigger_tokens)

if trigger_len > 0:
    for text in all_texts_raw:
        tokens = re.findall(r'\w+', text.lower())
        limit = len(tokens) - trigger_len - N_GRAM_SIZE
        
        for i in range(limit):
            # If the first word matches, check the rest
            if tokens[i] == trigger_tokens[0]:
                if tokens[i : i + trigger_len] == trigger_tokens:
                    # BINGO: Capture the next N words
                    seq = tokens[i + trigger_len : i + trigger_len + N_GRAM_SIZE]
                    found_sequences.append(" ".join(seq))

    # RESULTS 2
    print(f"\n--- TOP 20 SEQUENCES AFTER '{TRIGGER_PHRASE.upper()}' ---")
    if found_sequences:
        counts = Counter(found_sequences).most_common(20)
        for i, (seq, count) in enumerate(counts):
            print(f"{i+1:02d}. [{count:3d} times] {TRIGGER_PHRASE} {seq} ...")
    else:
        print("No sequences found.")
else:
    print("Error: Empty trigger.")