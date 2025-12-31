import pandas as pd
import re
import os
from collections import Counter
import random

# ==============================================================================
# ANALYSE TEXTUELLE concordance (SOURCE PARQUET) - OPTIMISÉE
# ==============================================================================

# --- 1. CONFIGURATION GLOBALE ---
PARQUET_FILES = [
    'DATA/CLEAN/PARQUET/qs_university_corpus.parquet',
    'DATA/CLEAN/PARQUET/the_university_corpus.parquet'
]

# Analyse 1 : Le mot pivot (ex: IMPACT)
TARGET_WORD = "impact"
CONTEXT_WINDOW = 6

# Analyse 2 : La phrase déclencheur (ex: AIMS TO)
TRIGGER_PHRASE = "aims to"
N_GRAM_SIZE = 6

# --- 2. CHARGEMENT DES DONNÉES (UNE SEULE FOIS) ---
print("=== 1. CHARGEMENT ET PRÉPARATION DES DONNÉES ===")

all_texts_raw = []

for f_path in PARQUET_FILES:
    print(f"-> Lecture de {os.path.basename(f_path)}...")
    try:
        df = pd.read_parquet(f_path)
        
        # Détection intelligente de la colonne texte
        text_col = None
        candidates = ['description', 'text', 'summary', 'overview', 'full_text']
        for col in df.columns:
            if col.lower() in candidates:
                text_col = col
                break
        
        if not text_col: # Fallback sur la 1ère colonne string
             text_col = df.select_dtypes(include=['object']).columns[0]

        print(f"   (Colonne utilisée : '{text_col}')")
        
        # On nettoie les NaN et on convertit tout en string
        texts = df[text_col].dropna().astype(str).tolist()
        all_texts_raw.extend(texts)
        
    except Exception as e:
        print(f"   /!\\ Erreur : {e}")

print(f"\n=> Corpus chargé : {len(all_texts_raw)} descriptions prêtes à l'analyse.")

# ==============================================================================
# ANALYSE 1 : CONTEXTE DU MOT (Ex: Quel type d'impact ?)
# ==============================================================================
print(f"\n" + "="*60)
print(f"=== 2. ANALYSE DU MOT CIBLE : '{TARGET_WORD.upper()}' ===")
print("="*60)

words_before_target = []
sample_sentences = []

for text in all_texts_raw:
    # Tokenisation simple (on garde les mots, on vire la ponctuation)
    tokens = re.findall(r'\w+', text.lower())
    
    if TARGET_WORD in tokens:
        indices = [i for i, x in enumerate(tokens) if x == TARGET_WORD]
        
        for i in indices:
            # A. Capture de l'adjectif juste avant (Analyse statistique)
            if i > 0:
                prev_word = tokens[i-1]
                stopwords = ["the", "a", "an", "of", "in", "to", "for", "and", "on", "with", "its", "their", "our", "have", "has"]
                if prev_word not in stopwords:
                    words_before_target.append(prev_word)
            
            # B. Capture de la phrase complète (Pour l'exemple)
            start = max(0, i - CONTEXT_WINDOW)
            end = min(len(tokens), i + CONTEXT_WINDOW + 1)
            snippet = tokens[start:end]
            # Mise en majuscule du mot cible
            snippet_str = " ".join([t.upper() if t == TARGET_WORD else t for t in snippet])
            sample_sentences.append(snippet_str)

# RÉSULTATS 1
print(f"\n--- TOP 15 ADJECTIFS AVANT '{TARGET_WORD.upper()}' ---")
if words_before_target:
    common = Counter(words_before_target).most_common(15)
    for word, freq in common:
        print(f"{word:>15}  {TARGET_WORD.upper()}  (x{freq})")
else:
    print("Pas assez de données.")

print(f"\n--- EXEMPLES DE CONTEXTE ---")
if sample_sentences:
    for s in random.sample(sample_sentences, min(5, len(sample_sentences))):
        print(f"... {s} ...")

# ==============================================================================
# ANALYSE 2 : N-GRAMMES (Ex: Que vise l'université ?)
# ==============================================================================
print(f"\n" + "="*60)
print(f"=== 3. ANALYSE DE SÉQUENCE : APRÈS '{TRIGGER_PHRASE.upper()}' ===")
print("="*60)

found_sequences = []
trigger_tokens = re.findall(r'\w+', TRIGGER_PHRASE.lower())
trigger_len = len(trigger_tokens)

if trigger_len > 0:
    for text in all_texts_raw:
        tokens = re.findall(r'\w+', text.lower())
        limit = len(tokens) - trigger_len - N_GRAM_SIZE
        
        for i in range(limit):
            # Si le premier mot matche, on vérifie la suite
            if tokens[i] == trigger_tokens[0]:
                if tokens[i : i + trigger_len] == trigger_tokens:
                    # BINGO : On prend les N mots suivants
                    seq = tokens[i + trigger_len : i + trigger_len + N_GRAM_SIZE]
                    found_sequences.append(" ".join(seq))

    # RÉSULTATS 2
    print(f"\n--- TOP 20 SUITES APRÈS '{TRIGGER_PHRASE.upper()}' ---")
    if found_sequences:
        counts = Counter(found_sequences).most_common(20)
        for i, (seq, count) in enumerate(counts):
            print(f"{i+1:02d}. [{count:3d} fois] {TRIGGER_PHRASE} {seq} ...")
    else:
        print("Aucune séquence trouvée.")
else:
    print("Erreur : Trigger vide.")