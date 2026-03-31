import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. LOAD CLEANED DATA ──────────────────────────────────────
print("Loading cleaned data...")
df = pd.read_csv('cleaned_data.csv')
df = df.dropna(subset=['clean_text'])  # remove any empty rows

X = df['clean_text']
y = df['target']

print(f"Total samples: {len(df)}")

# ── 2. VECTORIZE (text → numbers) ────────────────────────────
print("\nVectorizing text...")
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_vectorized = vectorizer.fit_transform(X)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

# ── 3. SPLIT DATA ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ── 4. TRAIN MODEL ────────────────────────────────────────────
print("\nTraining model... (2-3 mins)")
model = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
model.fit(X_train, y_train)
print("Training done!")

# ── 5. EVALUATE ───────────────────────────────────────────────
print("\nEvaluating...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# ── 6. CONFUSION MATRIX (save for report) ────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative','Positive'],
            yticklabels=['Negative','Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as confusion_matrix.png")

# ── 7. SAVE MODEL & VECTORIZER ────────────────────────────────
print("\nSaving model...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ model.pkl saved!")
print("✅ vectorizer.pkl saved!")
