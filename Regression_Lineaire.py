import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Chargement des données depuis le fichier CSV
datas = pd.read_csv('resultats_final.csv')

data = datas.copy()


# Affichage des premières lignes des données
print("Aperçu des données :")
print(data.head())
print(data.shape)

# Statistiques descriptives
print("\nStatistiques descriptives :")
print(data.describe())

# Sélection des variables numériques pour le calcul de la corrélation
data_numeric = data.select_dtypes(include=['float64', 'int64'])

# Matrice de corrélation
plt.figure(figsize=(10, 8))
corr_matrix = data_numeric.corr()
print("Matrice de corrélation :", corr_matrix)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de corrélation')
plt.show()

# # Visualisation des relations entre les variables
# plt.figure(figsize=(10, 6))
# sns.pairplot(data, hue='id_chapitre')
# plt.title('Diagramme de dispersion des variables')
# plt.show()

# selection des indices à supprimer (DureeTotal < 60000.0 et scrollMinute > 8) duree en millisecondes
indices = data[(data['DureeTotal'] < 60000.0) & (data['scrollMinute'] > 8.0)].index
# suppression des lignes avec les indices sélectionnés
data.drop(indices, inplace=True)

# selection des indices à supprimer (DureeTotal < 120000.0 et scrollMinute > 16) duree en millisecondes
indices = data[(data['DureeTotal'] < 120000.0) & (data['scrollMinute'] > 16.0)].index
# suppression des lignes avec les indices sélectionnés
data.drop(indices, inplace=True)

# selection des indices à supprimer (DureeTotal < 180000.0 et scrollMinute > 24) duree en millisecondes
indices = data[(data['DureeTotal'] < 180000.0) & (data['scrollMinute'] > 24.0)].index
# suppression des lignes avec les indices sélectionnés
data.drop(indices, inplace=True)

# selection des indices à supprimer (DureeTotal < 60000.0 et clicksMinute > 2) duree en millisecondes
indices = data[(data['DureeTotal'] < 60000.0) & (data['clicksMinute'] > 2.0)].index
# suppression des lignes avec les indices sélectionnés
data.drop(indices, inplace=True)

# selection des indices à supprimer (DureeTotal < 120000.0 et clicksMinute > 4) duree en millisecondes
indices = data[(data['DureeTotal'] < 120000.0) & (data['clicksMinute'] > 4.0)].index
# suppression des lignes avec les indices sélectionnés
data.drop(indices, inplace=True)

# selection des indices à supprimer (DureeTotal < 180000.0 et clicksMinute > 6) duree en millisecondes
indices = data[(data['DureeTotal'] < 180000.0) & (data['clicksMinute'] > 6.0)].index
# suppression des lignes avec les indices sélectionnés
data.drop(indices, inplace=True)

# selection des indices à supprimer
indices = data[(data['DureeTotal'] > 100000000)].index
# suppression des lignes avec les indices sélectionnés
data.drop(indices, inplace=True)

# selection des indices à supprimer
indices = data[(data['scrollMinute'] > 1) & (data['chapitreProgression'] < 0.1) ].index
# suppression des lignes avec les indices sélectionnés
data.drop(indices, inplace=True)

# selection des indices à supprimer
indices = data[(data['chapitreProgression'] < 50.00) & (data['note'] > 15.0) ].index
# suppression des lignes avec les indices sélectionnés
data.drop(indices, inplace=True)

# selection des indices à supprimer
indices = data[(data['scrollMinute'] < 5.0) & (data['chapitreProgression'] < 0.1) ].index
# suppression des lignes avec les indices sélectionnés
data.drop(indices, inplace=True)


# Statistiques descriptives apres suppression
print("\nStatistiques descriptives après suppression :")
print(data.describe())


# Sélection des variables numériques pour le calcul de la corrélation
data_numeric = data.select_dtypes(include=['float64', 'int64'])

# Matrice de corrélation
plt.figure(figsize=(10, 8))
corr_matrix = data_numeric.corr()
print("Matrice de corrélation :", corr_matrix)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de corrélation après suppression')
plt.show()

# Visualisation des distributions des variables numériques par domaine avec des boîtes à moustaches
plt.figure(figsize=(12, 8))
for i, feature in enumerate(['chapitreProgression', 'scrollMinute', 'clicksMinute', 'DureeTotal']):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='id_chapitre', y=feature, data=data)
    plt.title(f'Distribution de {feature} par chapitre')
    plt.xlabel('Chapitre')
    plt.ylabel(feature)
plt.tight_layout()
plt.show()




# suppression des valeurs aberrantes pour la variable 'DureeTotal' par chapitre
indices = data[(data['DureeTotal'] > 20000000)].index
data.drop(indices, inplace=True)




plt.figure(figsize=(10, 6))
sns.boxplot(x='id_chapitre', y='DureeTotal', data=data)
plt.title('Distribution de la durée totale par chapitre apres suppression des valeurs aberrantes')
plt.xlabel('Chapitre')
plt.ylabel('DureeTotal')
plt.show()



# Standardisation des variables numériques

# Création d'un objet de standardisation
scaler = StandardScaler()

# Sélection des caractéristiques à normaliser
features_to_normalize = ['chapitreProgression', 'scrollMinute', 'clicksMinute','note']

# Standardisation des caractéristiques sélectionnées
data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])

# Affichage des données standardisées
print(data.head())

# # suppression de la colonne clickMinute
# data.drop('clicksMinute', axis=1, inplace=True)

# Sélection des variables numériques pour le calcul de la corrélation
data_numeric = data.select_dtypes(include=['float64', 'int64'])

# Matrice de corrélation
plt.figure(figsize=(10, 8))
corr_matrix = data_numeric.corr()
print("Matrice de corrélation :", corr_matrix)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de corrélation après standardisation')
plt.show()

# Séparation des données en ensembles d'entraînement et de test
X = data[['chapitreProgression', 'scrollMinute','note','clicksMinute']]
y = data['DureeTotal']
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



