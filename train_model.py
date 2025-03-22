# train_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Dummy data for 20 stars (replace with your actual dataset)
data = {
    'period': [1.5, 2.0, 3.1, 0.5, 4.0, 2.5, 1.8, 3.0, 2.2, 1.0, 5.0, 2.8, 1.2, 3.5, 2.1, 1.7, 4.5, 2.3, 1.9, 3.2],
    'amplitude': [0.1, 0.2, 0.15, 0.05, 0.3, 0.12, 0.08, 0.25, 0.18, 0.07, 0.4, 0.14, 0.09, 0.22, 0.16, 0.11, 0.35, 0.13, 0.06, 0.28],
    'mean_flux': [1.0, 1.1, 0.9, 1.2, 0.95, 1.05, 1.0, 0.98, 1.02, 1.15, 0.92, 1.03, 1.08, 0.97, 1.04, 1.01, 0.96, 1.06, 1.07, 0.99],
    'std_flux': [0.02, 0.03, 0.01, 0.04, 0.05, 0.02, 0.03, 0.01, 0.04, 0.02, 0.06, 0.03, 0.02, 0.05, 0.03, 0.02, 0.04, 0.03, 0.02, 0.05],
    'skewness': [0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.1, -0.1, 0.2, -0.2, 0.25, -0.25, 0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.1, -0.1],
    'kurtosis': [3.0, 2.8, 3.1, 2.9, 3.2, 2.7, 3.0, 2.8, 3.1, 2.9, 3.3, 2.6, 3.0, 2.8, 3.1, 2.9, 3.2, 2.7, 3.0, 2.8],
    'max_power': [100, 150, 120, 80, 200, 110, 90, 180, 130, 70, 250, 140, 85, 170, 125, 95, 220, 115, 75, 190],
    'freq_at_max_power': [0.5, 0.4, 0.3, 0.6, 0.2, 0.45, 0.55, 0.35, 0.4, 0.65, 0.15, 0.42, 0.58, 0.38, 0.43, 0.52, 0.25, 0.48, 0.62, 0.32],
    'mad_flux': [0.01, 0.02, 0.015, 0.03, 0.025, 0.01, 0.02, 0.015, 0.03, 0.01, 0.035, 0.02, 0.01, 0.025, 0.015, 0.01, 0.03, 0.02, 0.01, 0.025],
    'num_peaks': [2, 3, 1, 4, 2, 3, 2, 1, 3, 2, 4, 2, 3, 1, 2, 3, 2, 1, 3, 2],
    'label': ['Eclipsing Binary', 'Eclipsing Binary', 'Transiting Exoplanet', 'Variable Star', 'Transiting Exoplanet',
              'Eclipsing Binary', 'Variable Star', 'Transiting Exoplanet', 'Eclipsing Binary', 'Variable Star',
              'Transiting Exoplanet', 'Eclipsing Binary', 'Variable Star', 'Transiting Exoplanet', 'Eclipsing Binary',
              'Variable Star', 'Transiting Exoplanet', 'Eclipsing Binary', 'Variable Star', 'Transiting Exoplanet']
}

# Create DataFrame
df = pd.DataFrame(data)
X = df.drop('label', axis=1)
y = df['label']

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Save the model
joblib.dump(clf, "star_classifier.pkl")
print("Model trained and saved as star_classifier.pkl")