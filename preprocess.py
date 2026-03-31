import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ── 1. LOAD DATASET ──────────────────────────────────────────
print("Loading dataset...")

df = pd.read_csv('train_data.csv', encoding='latin-1')
print("Columns found:", df.columns.tolist())

# Rename to standard names
df.columns = ['text', 'target']

# Convert labels: 4 → 1 (positive), 0 stays 0 (negative)
df['target'] = df['target'].replace(4, 1)

# Convert labels: 4 → 1 (positive), 0 stays 0 (negative)
df['target'] = df['target'].replace(4, 1)

print(f"Dataset loaded: {df.shape[0]} rows")
print(df['target'].value_counts())

# ── 2. CLEANING FUNCTION ─────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Keep sentiment words that stopwords would remove
keep_words = {'not', 'no', 'nor', 'but', 'very', 'too'}
stop_words = stop_words - keep_words

def clean_text(text):
    text = str(text).lower()                            # lowercase
    text = re.sub(r'http\S+|www\S+', '', text)         # remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)              # remove @mentions #hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)            # remove numbers & punctuation
    text = re.sub(r'\s+', ' ', text).strip()           # remove extra spaces
    
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    
    return ' '.join(tokens)

# ── 3. APPLY CLEANING ─────────────────────────────────────────
print("\nCleaning text... (this takes 3-5 mins, be patient)")

# Use only 200,000 rows to keep it fast
df = df.sample(n=200000, random_state=42).reset_index(drop=True)

df['clean_text'] = df['text'].apply(clean_text)

print("Cleaning done!")
print(df[['text', 'clean_text']].head(3))

# ── 4. SAVE CLEANED DATA ──────────────────────────────────────
df.to_csv('cleaned_data.csv', index=False)
print("\n✅ Saved as cleaned_data.csv")
