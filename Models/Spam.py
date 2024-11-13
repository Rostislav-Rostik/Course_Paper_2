import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# Load dataset
data = pd.read_csv("emails.csv")
X = data['text']
y = data['spam']  # Ensure this column name matches your dataset

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the email text with tuned parameters
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=2)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize a logistic regression model with balanced class weights
model = LogisticRegression(solver='saga', class_weight='balanced', C=0.5, max_iter=1000)

# Track indices of labeled data
labeled_indices = np.random.choice(X_train_tfidf.shape[0], size=10, replace=False)
unlabeled_indices = np.setdiff1d(np.arange(X_train_tfidf.shape[0]), labeled_indices)

# Train the model on the initial labeled data
model.fit(X_train_tfidf[labeled_indices], y_train.iloc[labeled_indices])

# Active learning loop
for iteration in range(20):  # Increasing iterations for more training opportunities
    # Predict probabilities for unlabeled samples
    probs = model.predict_proba(X_train_tfidf[unlabeled_indices])
    uncertainty = np.abs(probs[:, 1] - 0.5)  # Measure uncertainty
    
    # Select the top 5 most uncertain samples to label
    query_indices = np.argsort(uncertainty)[:5]
    labeled_indices = np.append(labeled_indices, unlabeled_indices[query_indices])
    unlabeled_indices = np.delete(unlabeled_indices, query_indices)
    
    # Train the model with the updated labeled set
    model.fit(X_train_tfidf[labeled_indices], y_train.iloc[labeled_indices])

    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"Iteration {iteration + 1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

# Final evaluation
y_pred = model.predict(X_test_tfidf)
print("Final Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")

# Save the model and vectorizer for use in a Web API
joblib.dump(model, 'spam_detection_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved successfully.")