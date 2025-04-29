import re
import pandas as pd
import numpy as np
import tldextract
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Function to extract URLs from email text
def extract_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.findall(text)

# Function to extract URL features
def extract_url_features(url):
    ext = tldextract.extract(url)
    return {
        "url_length": len(url),
        "num_dots": url.count('.'),
        "num_slashes": url.count('/'),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(c in "?=&" for c in url),
        "is_shortened": 1 if any(s in url for s in ["bit.ly", "tinyurl", "goo.gl"]) else 0,
        "has_ip": 1 if re.match(r"\d+\.\d+\.\d+\.\d+", ext.domain) else 0
    }

# Sample dataset (Replace with actual email dataset)
data = [
    {"text": "Click here to verify your account: https://secure-login.example.com", "label": "spam"},
    {"text": "Meeting at 5 PM, see details: https://company-internal.com/meeting", "label": "ham"},
    {"text": "You won a prize! Claim now: http://win-free-money.com", "label": "spam"},
    {"text": "Project update: www.workplace-news.com/update", "label": "ham"}
]

df = pd.DataFrame(data)

# Extract URLs and their features
df["urls"] = df["text"].apply(extract_urls)
df = df.explode("urls").reset_index(drop=True)
df.dropna(subset=["urls"], inplace=True)

# Extract URL-based features
url_features = df["urls"].apply(lambda x: extract_url_features(x))
url_features_df = pd.DataFrame(url_features.tolist())
df = pd.concat([df, url_features_df], axis=1)

# Text vectorization
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(df["text"])

# Combine text and URL-based features
X_combined = np.hstack((X_text.toarray(), df[url_features_df.columns].values))

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")