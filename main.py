# Bibliotheken zur Datenmanipulation und -analyse
import pandas as pd, numpy as np
# Visualisierung der Daten
import matplotlib.pyplot as plt, seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Daten einlesen und vorverarbeiten
stress_data = pd.read_csv('data.csv', engine='python' ,parse_dates=['Date of Joining'])
stress_data = stress_data.dropna()
stress_data = stress_data.drop("Employee ID", axis=1)

# Funktion zur Visualisierung von Streudiagrammen
def visualize_scatter(data, x, y, hue=None, color=None, title=None):
    sns.scatterplot(data=data, x=x, y=y, hue=hue, color=color)
    plt.title(title)
    plt.show()

# Funktion zur Visualisierung von Streudiagrammen mit Linearer Regression
def visualize_scatter_with_regression(data, x, y, color, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, data=data, color=color)
    sns.regplot(x=x, y=y, data=data, scatter=False, color='red', line_kws={'label': 'Lineare Regression'})
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

# Funktion zum Binarisieren bestimmter Spalten  
def binarize_columns(data, columns):
    return pd.get_dummies(data, columns=columns, drop_first=True)

# Funktion zur Bewertung des Modells
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, mae


# Visualisierung der Daten
visualize_scatter(stress_data, 'Mental Fatigue Score', 'Burn Rate', title='Burn Rate x Mental Fatigue Score')
visualize_scatter(stress_data, 'Mental Fatigue Score', 'Burn Rate', hue='WFH Setup Available', title='Burn Rate x Mental Fatigue Score with WFH Setup Available')
visualize_scatter(stress_data, 'Date of Joining', 'Burn Rate', title='Burn Rate x Date of Joining')
visualize_scatter(stress_data, 'Designation', 'Burn Rate', title='Burn Rate x Designation')
visualize_scatter(stress_data, 'Resource Allocation', 'Burn Rate', title='Burn Rate x Resource Allocation')

male_data = stress_data[stress_data['Gender'] == 'Male']
female_data = stress_data[stress_data['Gender'] == 'Female']

# Streudiagramm für Männer
visualize_scatter(male_data, 'Mental Fatigue Score', 'Burn Rate', title='Streudiagramm für Männer')
# Streudiagramm für Frauen
visualize_scatter(female_data, 'Mental Fatigue Score', 'Burn Rate', color='green', title='Streudiagramm für Frauen')

# Entfernen der Spalte 'Date of Joining'
stress_data = stress_data.drop('Date of Joining', axis=1)

# Binarisierung
binary_cols = ['Gender', 'Company Type', 'WFH Setup Available']
stress_data = binarize_columns(stress_data, binary_cols)

# Aufteilung in X und y
y = stress_data['Burn Rate']
X = stress_data.drop('Burn Rate', axis=1)

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)

# Skalierung von X
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
r2_scores = []

# Instanzierung und Anpassen des Modells
lm = LinearRegression().fit(X_train_scaled, y_train)

# Vorhersagen
lm_y_pred = lm.predict(X_test_scaled)

# Auswertung des linearen Regressionsmodells
lr_r2, lr_mae = evaluate_model(y_test, lm_y_pred)

# Anzeigen der Metriken für das lineare Regressionsmodell
print("Linear Regression R2: ", lr_r2)
print("Linear Regression MAE: ", lr_mae)

# Modelle für die Regression in einem Tupel
models = [
    ("Linear Regression", LinearRegression()),
    ("Linear Regression (L2 Regularization)", Ridge()),
    ("K-Nearest Neighbors", KNeighborsRegressor()),
    ("Neural Network", MLPRegressor()),
    ("Support Vector Machine (Linear Kernel)", LinearSVR(max_iter=10000, tol=1e-5, dual=False, loss='squared_epsilon_insensitive')),
    ("Support Vector Machine (RBF Kernel)", SVR()),
    ("Decision Tree", DecisionTreeRegressor()),
    ("Random Forest", RandomForestRegressor()),
    ("Gradient Boosting", GradientBoostingRegressor())
]

# Erstellen eines leeren Dictionary zum Speichern der Ergebnisse
results_dict = {'Model': [], 'R2 Score': []}

# Schleife für Modelltraining und Bewertung
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{name} - R2 Score: {r2:.4f}, MAE: {mae:.4f}")
    results_dict['Model'].append(name)
    results_dict['R2 Score'].append(r2)


# Konvertieren des Dictionary in ein DataFrame
results_df = pd.DataFrame(results_dict)

# Visualisierung der R2-Scores
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='R2 Score', data=results_df)
plt.title('R2 Scores der verschiedenen Modelle')
plt.ylabel('R2 Score')
plt.xticks(rotation=45, ha='right')
plt.show()

# Visualisierung für Männer
visualize_scatter_with_regression(male_data, 'Mental Fatigue Score', 'Burn Rate', 'blue', 'Streudiagramm für Männer mit Linearer Regression')

# Visualisierung für Frauen
visualize_scatter_with_regression(female_data, 'Mental Fatigue Score', 'Burn Rate', 'green', 'Streudiagramm für Frauen mit Linearer Regression')
