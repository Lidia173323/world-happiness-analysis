import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie pliku CSV
df = pd.read_csv('world_happiness_report.csv')

# Wyświetlenie pierwszych 5 wierszy
print("--- Pierwsze 5 wierszy danych ---")
print(df.head())

# Wyświetlenie informacji o kolumnach i typach danych
print("\n--- Informacje o danych (typach i brakujących wartościach) ---")
print(df.info())

# Statystyki opisowe
print("\n--- Statystyki opisowe (zmienne numeryczne) ---")
print(df.describe().T)

print("\n--- Statystyki opisowe (zmienne obiektowe) ---")
print(df.describe(include='object').T)

# Obliczenie procentu brakujących wartości
missing_data = df.isnull().sum().sort_values(ascending=False)
missing_percentage = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing_df = pd.DataFrame({'Missing Count': missing_data, 'Missing Percentage': missing_percentage})
missing_df = missing_df[missing_df['Missing Count'] > 0]

print("\n--- Brakujące Wartości w Danych ---")
print(missing_df)

# Sprawdzenie duplikatów
print(f"\nLiczba duplikatów: {df.duplicated().sum()}")

# Analiza lat
print("\n--- Rozkład danych według roku ---")
print(df['year'].value_counts())

# Filtracja danych do lat 2015-2016
print("\n--- Brakujące wartości dla lat 2015-2016 ---")
print(df[df['year'].isin([2015, 2016])].isna().sum())

df = df[df['year'].isin([2015, 2016])]
print(f"\nLiczba rekordów po filtracji: {len(df)}")

# Definiowanie kluczowych kolumn
key_columns = ['Happiness Score', 'Economy (GDP per Capita)', 'Family',
               'Health (Life Expectancy)', 'Freedom',
               'Trust (Government Corruption)', 'Dystopia Residual']

print("\n--- Statystyki Opisowe (Kluczowe Zmienne) ---")
print(df[key_columns].describe())

# Wizualizacja rozkładu Happiness Score
plt.figure(figsize=(10, 6))
sns.histplot(df['Happiness Score'], color='purple')
plt.title('Rozkład Happiness Score')
plt.xlabel('Happiness Score')
plt.ylabel('Częstość')
plt.show()

# Boxplot według roku
plt.figure(figsize=(15, 10))
sns.boxplot(data=df, x='year', y='Happiness Score', palette='magma')
plt.xlabel('Year')
plt.ylabel('Happiness Score')
plt.title('Happiness Score według roku')
plt.show()

# Pairplot
sns.pairplot(df[key_columns])
plt.show()

# Heatmapa korelacji - wszystkie dane
plt.figure(figsize=(10, 6))
dcorr = df[key_columns].select_dtypes(include='float64').corr()
sns.heatmap(dcorr, annot=True, cmap='coolwarm')
plt.title('Macierz Korelacji - Wszystkie Dane')
plt.show()

correlation_matrix = df[key_columns].corr()

# Wyświetlenie korelacji z Happiness Score
print("\n--- Korelacja Czynników z Happiness Score ---")
print(correlation_matrix['Happiness Score'].sort_values(ascending=False))

# Heatmapa korelacji - rok 2015
plt.figure(figsize=(10, 6))
dcorr = df[df['year'] == 2015][key_columns].select_dtypes(include='float64').corr()
sns.heatmap(dcorr, annot=True, cmap='coolwarm')
plt.title('Macierz Korelacji - Rok 2015')
plt.show()

# Heatmapa korelacji - rok 2016
plt.figure(figsize=(10, 6))
dcorr = df[df['year'] == 2016][key_columns].select_dtypes(include='float64').corr()
sns.heatmap(dcorr, annot=True, cmap='coolwarm')
plt.title('Macierz Korelacji - Rok 2016')
plt.show()

# ==================== PRZYGOTOWANIE DANYCH ====================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Wybór cech (X) i zmiennej docelowej (y)
X = df[key_columns].drop('Happiness Score', axis=1)
y = df['Happiness Score']

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nRozmiar zbioru treningowego (X_train): {X_train.shape}")
print(f"Rozmiar zbioru testowego (X_test): {X_test.shape}")

# ==================== MODEL 1: LINEAR REGRESSION ====================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Standaryzacja danych dla regresji liniowej
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inicjalizacja i trening modelu
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predykcje na zbiorze testowym
lr_pred = lr_model.predict(X_test_scaled)

# Ocena modelu
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("\n" + "="*60)
print("--- Wyniki Regresji Liniowej ---")
print("="*60)
print(f"R² (Współczynnik Determinacji): {lr_r2:.4f}")
print(f"RMSE (Root Mean Squared Error): {lr_rmse:.4f}")

# Walidacja krzyżowa (cv=5)
lr_cv = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"Cross-val R²: {lr_cv.mean():.4f}")

# ==================== MODEL 2: RANDOM FOREST ====================

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Definicja modelu bazowego
rf = RandomForestRegressor(random_state=42)

# Siatka hiperparametrów
param_grid = {
    'n_estimators': [100, 300, 500, 600],
    'max_depth': [None, 5, 10, 20, 25],
    'min_samples_split': [2, 5, 7, 10]
}

# Grid Search
print("\n" + "="*60)
print("--- Random Forest - Grid Search ---")
print("="*60)
print("Rozpoczęcie Grid Search (może potrwać kilka minut)...")

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# Trenowanie Grid Search
grid_search.fit(X_train, y_train)

# Najlepszy model
best_rf = grid_search.best_estimator_

print("\nNajlepsze parametry Grid Search:")
print(grid_search.best_params_)

# Predykcje na zbiorze testowym
rf_pred = best_rf.predict(X_test)

# Ocena modelu
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n--- Wyniki Random Forest (po Grid Search) ---")
print(f"R² (Współczynnik Determinacji): {rf_r2:.4f}")
print(f"RMSE (Root Mean Squared Error): {rf_rmse:.4f}")

# Walidacja krzyżowa
rf_cv = cross_val_score(best_rf, X, y, cv=5, scoring='r2')
print(f"Cross-val R²: {rf_cv.mean():.4f}")

# Ważność cech z modelu Random Forest
feature_importances = pd.Series(best_rf.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

print("\n--- Ważność Cech w Modelu Random Forest ---")
print(feature_importances)

plt.figure(figsize=(12, 7))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="viridis", hue=feature_importances.index, legend=False)
plt.title('Ważność Czynników w Predykcji Szczęścia (Model Random Forest)')
plt.xlabel('Ważność (Wkład w redukcję błędu)')
plt.ylabel('Czynnik')
plt.tight_layout()
plt.show()

# ==================== MODEL 3: XGBOOST ====================

import xgboost as xgb

# Inicjalizacja modelu XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Siatka hiperparametrów
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.07, 0.1],
    'subsample': [0.7, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0]
}

# Grid Search
print("\n" + "="*60)
print("--- XGBoost - Grid Search ---")
print("="*60)
print("Rozpoczęcie Grid Search (może potrwać kilka minut)...")

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Trenowanie Grid Search
grid_search.fit(X_train, y_train)

# Najlepszy model
best_xgb = grid_search.best_estimator_

print("\nNajlepsze parametry Grid Search:")
print(grid_search.best_params_)

# Predykcje na zbiorze testowym
xgb_pred = best_xgb.predict(X_test)

# Ocena modelu
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print("\n--- Wyniki XGBoost (po Grid Search) ---")
print(f"R² (Współczynnik Determinacji): {xgb_r2:.4f}")
print(f"RMSE (Root Mean Squared Error): {xgb_rmse:.4f}")

# Walidacja krzyżowa
xgb_cv = cross_val_score(best_xgb, X, y, cv=5, scoring='r2')
print(f"Cross-val R²: {xgb_cv.mean():.4f}")

# Ważność cech
xgb_feature_importances = pd.Series(best_xgb.feature_importances_, index=X.columns)
xgb_feature_importances = xgb_feature_importances.sort_values(ascending=False)

print("\n--- Ważność Cech w Modelu XGBoost ---")
print(xgb_feature_importances)

plt.figure(figsize=(12, 7))
sns.barplot(x=xgb_feature_importances.values, y=xgb_feature_importances.index, palette="viridis", hue=xgb_feature_importances.index, legend=False)
plt.title('Ważność Cech w Predykcji Szczęścia (Model XGBoost)')
plt.xlabel('Ważność (Wkład w redukcję błędu)')
plt.ylabel('Czynnik')
plt.tight_layout()
plt.show()

# ==================== MODEL 4: GRADIENT BOOSTING ====================

from sklearn.ensemble import GradientBoostingRegressor

# Definicja modelu Gradient Boosting
gbr_model = GradientBoostingRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.9, 1.0],
    'min_samples_split': [2, 5, 10]
}

print("\n" + "="*60)
print("--- Gradient Boosting - Grid Search ---")
print("="*60)
print("Rozpoczęcie Grid Search (może potrwać kilka minut)...")

grid_search = GridSearchCV(
    estimator=gbr_model,
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_gbr = grid_search.best_estimator_

print("\nNajlepsze parametry Grid Search:")
print(grid_search.best_params_)

gbr_pred = best_gbr.predict(X_test)

gbr_rmse = np.sqrt(mean_squared_error(y_test, gbr_pred))
gbr_r2 = r2_score(y_test, gbr_pred)

print("\n--- Gradient Boosting Regressor (po Grid Search) ---")
print(f"R²: {gbr_r2:.4f}")
print(f"RMSE: {gbr_rmse:.4f}")

cv_scores = cross_val_score(best_gbr, X, y, cv=5, scoring='r2')
print(f"Cross-val R²: {cv_scores.mean():.4f}")

gbr_feature_importances = pd.Series(best_gbr.feature_importances_, index=X.columns)
gbr_feature_importances = gbr_feature_importances.sort_values(ascending=False)

print("\n--- Ważność Cech w Modelu Gradient Boosting ---")
print(gbr_feature_importances)

plt.figure(figsize=(12, 7))
sns.barplot(x=gbr_feature_importances.values, y=gbr_feature_importances.index, palette="viridis", hue=gbr_feature_importances.index, legend=False)
plt.title('Ważność Cech w Predykcji Szczęścia (Model Gradient Boosting)')
plt.xlabel('Ważność (Wkład w redukcję błędu)')
plt.ylabel('Czynnik')
plt.tight_layout()
plt.show()

# ==================== PORÓWNANIE MODELI ====================

results = {
    'Linear Regression': {'R2': lr_r2, 'RMSE': lr_rmse, 'CV R2': lr_cv.mean()},
    'Random Forest': {'R2': rf_r2, 'RMSE': rf_rmse, 'CV R2': rf_cv.mean()},
    'XGBoost': {'R2': xgb_r2, 'RMSE': xgb_rmse, 'CV R2': xgb_cv.mean()},
    'Gradient Boosting': {'R2': gbr_r2, 'RMSE': gbr_rmse, 'CV R2': cv_scores.mean()}
}

results_df = pd.DataFrame(results).T.sort_values(by='R2', ascending=False)

print("\n" + "="*60)
print("--- PORÓWNANIE WSZYSTKICH MODELI ---")
print("="*60)
print(results_df)

# Wizualizacja porównania R² i RMSE
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

results_df['R2'].plot(kind='bar', ax=ax[0], title='Porównanie R² Modeli (Im wyżej, tym lepiej)', 
                       color=['skyblue', 'lightgreen', 'orange', 'salmon'])
ax[0].set_ylabel('R² Score')
ax[0].tick_params(axis='x', rotation=45)
ax[0].grid(axis='y', alpha=0.3)

results_df['RMSE'].plot(kind='bar', ax=ax[1], title='Porównanie RMSE Modeli (Im niżej, tym lepiej)', 
                         color=['skyblue', 'lightgreen', 'orange', 'salmon'])
ax[1].set_ylabel('RMSE Score')
ax[1].tick_params(axis='x', rotation=45)
ax[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== ANALIZA KRAJÓW ====================

# Usunięcie brakujących wartości w kolumnie Country
df['Country'].isna().sum()
df.dropna(subset=['Country'], inplace=True)

# Obliczenie średniego poziomu szczęścia według kraju
happiness_by_country = df.groupby('Country')['Happiness Score'].mean().sort_values(ascending=False)

print("\n" + "="*60)
print("--- ANALIZA KRAJÓW ---")
print("="*60)

print("\n--- Top 5 Krajów z Największym Poziomem Szczęścia ---")
print(happiness_by_country.head(5))

# Wizualizacja top 15 najszczęśliwszych krajów
plt.figure(figsize=(20, 10))
sns.barplot(x=happiness_by_country.head(15).index, y=happiness_by_country.head(15).values, palette='magma')
plt.title('Top 15 Krajów z Największym Poziomem Szczęścia', fontsize=16, fontweight='bold')
plt.xlabel('Kraj', fontsize=14)
plt.ylabel('Średni Poziom Szczęścia', fontsize=14)
plt.xticks(rotation=45, ha='right')
for index, value in enumerate(happiness_by_country.head(15).values):
    plt.text(index, value + 0.05, f'{value:.2f}', color='black', ha='center')
plt.tight_layout()
plt.show()

print("\n--- Top 5 Krajów z Najniższym Poziomem Szczęścia ---")
print(happiness_by_country.tail(5))

# Wizualizacja top 15 najmniej szczęśliwych krajów
plt.figure(figsize=(20, 10))
sns.barplot(x=happiness_by_country.tail(15).index, y=happiness_by_country.tail(15).values, palette='magma')
plt.title('Top 15 Krajów z Najniższym Poziomem Szczęścia', fontsize=16, fontweight='bold')
plt.xlabel('Kraj', fontsize=14)
plt.ylabel('Średni Poziom Szczęścia', fontsize=14)
plt.xticks(rotation=45, ha='right')
for index, value in enumerate(happiness_by_country.tail(15).values):
    plt.text(index, value + 0.05, f'{value:.2f}', color='black', ha='center')
plt.tight_layout()
plt.show()

# ==================== MAPA ŚWIATA ====================

import plotly.express as px

fig = px.choropleth(df, 
                    locations='Country', 
                    locationmode='country names',
                    color='Happiness Score',
                    hover_name='Country',
                    hover_data=['Happiness Score'],
                    color_continuous_scale='Viridis',
                    title='Mapa Poziomu Szczęścia na Świecie (2015-2016)')
fig.show()

print("\n" + "="*60)
print("ANALIZA ZAKOŃCZONA POMYŚLNIE!")
print("="*60)
