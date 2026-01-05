import re
import enchant
import pandas as pd
from tqdm import tqdm

tqdm.pandas() #activate progress_apply

print("Check if the enchant_file path is written [...]enchant\__init__.py")
print(enchant.__file__)
print("This python needs to be interpreted by an python version below 3.13")

# Load the English dictionary
checker = enchant.Dict("en_US")

def split_merged_word(word):
    """
    Try to split a merged word into two valid English words.
    Only split if the original word is NOT already a valid word.
    """
    # 1. If the word is already valid English, return it unchanged
    if checker.check(word):
        return word, False

    # 2. Otherwise, try to split it into two valid words
    word_lower = word.lower()

    # Try all possible split positions, from index 2 to len(word)-2
    for i in range(2, len(word) - 2): #start at 2 and finish at len -2 because a word is minimum 2 characters
        left = word_lower[:i]   # left part of the word
        right = word_lower[i:]  # right part of the word

        # Check if both parts are valid English words
        if checker.check(left) and checker.check(right):
            # Preserve capitalization for the first letter if needed
            if word[0].isupper():
                return f"{left.capitalize()} {right}",True
            return f"{left} {right}",True

    # 3. If no valid split is found, return the original word
    return word,False


def clean_text(text):
    """
    Fix merged words and add spaces after punctuation when needed.
    This function only modifies words that are not valid English words.
    """
    if text is None:
        return text,0

    n_changes = 0 # local counter for this text

    # Inner function used by re.sub to process each matched word
    def replace_word(match):
        nonlocal n_changes
        # match.group(0) is the full matched word
        word = match.group(0)
        new_word, changed = split_merged_word(word)
        if changed :
            n_changes +=1
        return new_word

    # 1. Apply the correction token by token, allowing internal hyphens "-"
    text = re.sub(r'\b[a-zA-Z]+(?:-[a-zA-Z]+)*\b', replace_word, text)

    #\b is a word boudnary the start or the end
    #[a-zA-Z]+ is the alphabetic part (no digits)
    # (?:-[a-zA-Z]+)* allows zero or more internanl hyphens,to catch hyphen zero or multiple times with * and (?: ... ) is a non-capturing group

    # 2. Add a space after punctuation when it is directly followed by a letter
    #    Example: ",founded" -> ", founded"

    text = re.sub(r'([.,!?;:])([a-zA-Z])', r'\1 \2', text)
    #[.,!?;:] capture the punctuation used as \1 afterwards
    #[a-zA-Z] capture the frist letter that follows directly without space
    #r  = raw string without "\" 
    #.sub means subsitytion re.sub(pattern, replacement, string) pattern = regex rule replacement = what to replace it with and string = the text to modify
    
    return text, n_changes

# Test
text = """The Université catholique de Louvain,founded in 1425, is a comprehensive 
university ofBelgium. Its 22 researchinstitutes, 40 technology platforms, and 3 scienceparks, 
which include 2 incubators and host more than 280 companies and 84spin-offs and start-ups, 
are a testament to UCLouvain's dedication toconducting fundamental and applied research. 
They also, along with UCLouvain'sfaculties and schools, exercise a cross-disciplinary research 
approach, itselffounded on a rich research tradition personified by such former professors 
asGeorges Lemaître, the father of the Big Bang theory, and Christian de Duve, winnerof 
the Nobel Prize in Medicine."""

#cleaned = clean_text(text)
#print(cleaned)


df = pd.read_parquet("DATA/CLEAN/PARQUET/qs_university_corpus_no_cleaned_description.parquet")

# Apply cleaning on the 'description' column
results = df["description"].progress_apply(clean_text) 

#result is a tuple (cleaned_text, n_changes)
#so the tuple needs to be split into the description and the number of changes through a lambda function
# lambda function are temporary and anonymous function to extract the first data in the tuple or the second. 
df["description"] = results.apply(lambda x: x[0])
df["n_changes"] = results.apply(lambda x: x[1])

# Total number of changed words across all descriptions
total_changes = df["n_changes"].sum()
print(f"Total number of corrected merged words: {total_changes}")

#reorganising the order in the document

base_order = ["name", "url", "rank", "region", "country"]

# new order : name, url, rank, region, n_changes, description, puis le reste
new_cols = base_order + ["n_changes", "description"]

df = df[new_cols]

# Save the cleaned data
df.to_parquet("DATA/CLEAN/PARQUET/qs_university_corpus.parquet", index=False)
df.to_csv("DATA/CLEAN/CSV/qs_university_corpus.csv", index=False)
