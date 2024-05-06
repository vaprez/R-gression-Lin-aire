import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Chargement des données depuis le fichier CSV
datas = pd.read_csv('donnees_quiz.csv')

data = datas.copy()

# Affichage des premières lignes des données
print("Aperçu des données :")
print(data.head())

# Statistiques descriptives
print("\nStatistiques descriptives :")
print(data.describe())

# # Visualisation de la distribution des variables numériques
# plt.figure(figsize=(10, 6))
# sns.histplot(data['Temps_Lecture'], kde=True, color='blue', bins=30)
# plt.title('Distribution du temps de lecture')
# plt.xlabel('Temps de lecture (minutes)')
# plt.ylabel('Fréquence')
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.histplot(data['Scrolls'], kde=True, color='green', bins=30)
# plt.title('Distribution du nombre de scrolls')
# plt.xlabel('Nombre de scrolls')
# plt.ylabel('Fréquence')
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.histplot(data['Clics'], kde=True, color='red', bins=30)
# plt.title('Distribution du nombre de clics')
# plt.xlabel('Nombre de clics')
# plt.ylabel('Fréquence')
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.histplot(data['Progression'], kde=True, color='purple', bins=30)
# plt.title('Distribution de la progression')
# plt.xlabel('Progression')
# plt.ylabel('Fréquence')
# plt.show()

# Visualisation des relations entre les variables
plt.figure(figsize=(10, 6))
sns.pairplot(data, hue='Domaine')
plt.title('Diagramme de dispersion des variables')
plt.show()


# Sélection des variables numériques pour le calcul de la corrélation
data_numeric = data.select_dtypes(include=['float64', 'int64'])

# Matrice de corrélation
plt.figure(figsize=(10, 8))
corr_matrix = data_numeric.corr()
print("Matrice de corrélation :", corr_matrix)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de corrélation')
plt.show()


# Sélection des indices des lignes à supprimer
# Pas normal d'avoir une progression < 0.5 et une note de quiz > 15
indices = data[(data['Note_Quiz'] > 15.0) & (data['Progression'] < 0.5)].index
print ("les indices",indices)
data.drop(indices, inplace=True)


# Pas normal d'avoir une note de quiz entre 15 et 18 avec un temps de lecture < 20 minutes
indices2 = data[(data['Note_Quiz'] > 15.0) & (data['Note_Quiz'] < 18.0) & (data['Temps_Lecture'] < 20)].index
print ("les indices2",indices2)

# Suppression des lignes correspondantes

data.drop(indices2, inplace=True)

# Affichage des données mises à jour
print("Aperçu des données après correction :")
print(data.head())


# Visualisation des relations entre les variables
plt.figure(figsize=(10, 6))
sns.pairplot(data, hue='Domaine')
plt.title('Diagramme de dispersion des variables apres regularisation progression et note')
plt.show()

# Statistiques descriptives
print("\nStatistiques descriptives :")
print(data.describe())


# Création de sous-ensembles de données basés sur la variable catégorielle 'Domaine'
grouped_data = data.groupby('Domaine')

# Analyse des différences entre les groupes
for group_name, group_data in grouped_data:
    print(f"Analyse pour le groupe '{group_name}':")
    
    # Statistiques descriptives pour chaque groupe
    print(group_data.describe())
    
    # Visualisation des distributions des variables numériques pour chaque groupe
    plt.figure(figsize=(10, 6))
    sns.histplot(group_data['Note_Quiz'], kde=True, color='blue', bins=30)
    plt.title(f'Distribution des notes pour le groupe "{group_name}"')
    plt.xlabel('Note au quiz')
    plt.ylabel('Fréquence')
    plt.show()

data_numeric = data.select_dtypes(include=['float64', 'int64'])

# Calcul de la matrice de corrélation
correlation_matrix = data_numeric.corr()

# Visualisation de la matrice de corrélation avec un heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matrice de corrélation entre les caractéristiques avant standardisation')
plt.show()

# Standardisation des variables numériques

# Création d'un objet de standardisation
scaler = StandardScaler()

# Sélection des caractéristiques à normaliser
features_to_normalize = ['Note_Quiz', 'Scrolls', 'Clics', 'Progression']

# Standardisation des caractéristiques sélectionnées
data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])

# Affichage des données standardisées
print(data.head())


data_numeric = data.select_dtypes(include=['float64', 'int64'])

# Calcul de la matrice de corrélation
correlation_matrix = data_numeric.corr()

# Visualisation de la matrice de corrélation avec un heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matrice de corrélation entre les caractéristiques apres standardisation')
plt.show()


# Séparation des données en ensembles d'entraînement et de test
X = data[['Note_Quiz', 'Scrolls', 'Clics', 'Progression']]
y = data['Temps_Lecture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Coefficient de détermination (R²) : {r2}")
print(f"RMSE (Root Mean Squared Error) : {rmse}")


# Visualisation des distributions des variables numériques par domaine avec des boîtes à moustaches
plt.figure(figsize=(12, 8))
for i, feature in enumerate(['Note_Quiz', 'Scrolls', 'Clics', 'Progression']):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='Domaine', y=feature, data=data)
    plt.title(f'Distribution de {feature} par domaine')
    plt.xlabel('Domaine')
    plt.ylabel(feature)
plt.tight_layout()
plt.show()





