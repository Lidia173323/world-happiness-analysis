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

# Obliczenie procentu brakujących wartości
missimg_data = missing_data = df.isnull().sum().sort_values(ascending=False)
missing_percentage = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing_df = pd.DataFrame({'Missing Count': missing_data, 'Missing Percentage': missing_percentage})
missing_df = missing_df[missing_df['Missing Count'] > 0]

print("\n--- Brakujące Wartości w Danych ---")
print(missing_df)

df.duplicated().sum()

df['year'].value_counts()

df[df['year'].isin([2015, 2016])].isna().sum()

df = df[df['year'].isin([2015, 2016])]

len(df)

key_columns = ['Happiness Score', 'Economy (GDP per Capita)', 'Family',
               'Health (Life Expectancy)', 'Freedom',
               'Trust (Government Corruption)', 'Dystopia Residual']

print("\n--- Statystyki Opisowe ---")
print(df[key_columns].describe())

plt.figure(figsize=(10, 6))
sns.histplot(df['Happiness Score'], kde=True, color='purple')

plt.figsize = (15, 10)
sns.boxplot(data=df, x='year', y='Happiness Score', palette='magma')
plt.xlabel('Year')

sns.pairplot(df[key_columns])

plt.figure(figsize=(10, 6))
dcorr = df[key_columns].select_dtypes(include='float64').corr()
sns.heatmap(dcorr, annot=True, cmap='coolwarm')

correlation_matrix = df[key_columns].corr()

# Wyświetlenie korelacji z Happiness Score
print("--- Korelacja Czynników z Happiness Score ---")
print(correlation_matrix['Happiness Score'].sort_values(ascending=False))

plt.figure(figsize=(10, 6))
dcorr = df[df['year'] == 2015][key_columns].select_dtypes(include='float64').corr()
sns.heatmap(dcorr, annot=True, cmap='coolwarm')

plt.figure(figsize=(10, 6))
dcorr = df[df['year'] == 2016][key_columns].select_dtypes(include='float64').corr()
sns.heatmap(dcorr, annot=True, cmap='coolwarm')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Wybór cech (X) i zmiennej docelowej (Y)
X_nor = df[key_columns].drop('Happiness Score', axis=1)
y = df['Happiness Score']

# Standardyzacja danych
scaler = StandardScaler()
X = scaler.fit_transform(X_nor)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Rozmiar zbioru treningowego (X_train): {X_train.shape}")
print(f"Rozmiar zbioru testowego (X_test): {X_test.shape}")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Inicjalizacja i trening modelu
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predykcje na zbiorze testowym
lr_pred = lr_model.predict(X_test)

# Ocena modelu
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("--- Wyniki Regresji Liniowej ---")
print(f"R^2 (Współczynnik Determinacji): {lr_r2:.4f}")
print(f"RMSE (Root Mean Squared Error): {lr_rmse:.4f}")

# Walidacja krzyżowa (cv=5)
lr_cv = cross_val_score(lr_model, X, y, cv=5, scoring='r2')
print("Cross-val R²:", lr_cv.mean().round(4))

from sklearn.ensemble import RandomForestRegressor

# Inicjalizacja i trening modelu
rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
rf_model.fit(X_train, y_train)

# Predykcje na zbiorze testowym
rf_pred = rf_model.predict(X_test)

# Ocena modelu
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n--- Wyniki Random Forest Regressor ---")
print(f"R^2 (Współczynnik Determinacji): {rf_r2:.4f}")
print(f"RMSE (Root Mean Squared Error): {rf_rmse:.4f}")

rf_cv = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print("Cross-val R²:", rf_cv.mean().round(4))

# Ważność cech z modelu Random Forest
feature_importances = pd.Series(rf_model.feature_importances_, index=X_nor.columns)

# Sortowanie i wizualizacja
feature_importances = feature_importances.sort_values(ascending=False)

print("\n--- Ważność Cech w Modelu Random Forest ---")
print(feature_importances)

plt.figure(figsize=(12, 7))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="viridis", hue=feature_importances.index, legend=False)
plt.title('Ważność Czynników w Predykcji Szczęścia (Model Random Forest)')
plt.xlabel('Ważność (Wkład w redukcję błędu)')
plt.ylabel('Czynnik')
plt.show()

import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Inicjalizacja i trening modelu XGBoost
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.07,
    random_state=42
)

# Trening na zbiorze treningowym
xgb_model.fit(X_train, y_train)

# Predykcje na zbiorze testowym
xgb_pred = xgb_model.predict(X_test)

# Ocena modelu
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print("--- Wyniki XGBoost Regressor ---")
print(f"R^2 (Współczynnik Determinacji): {xgb_r2:.4f}")
print(f"RMSE (Root Mean Squared Error): {xgb_rmse:.4f}")

xgb_cv = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')
print("Cross-val R²:", xgb_cv.mean().round(4))

xgb_feature_importances = pd.Series(xgb_model.feature_importances_, index=X_nor.columns)
xgb_feature_importances = xgb_feature_importances.sort_values(ascending=False)

print("\n--- Ważność Cech w Modelu XGBoost ---")
print(xgb_feature_importances)

plt.figure(figsize=(12, 7))
sns.barplot(x=xgb_feature_importances.values, y=xgb_feature_importances.index, palette="viridis", hue=xgb_feature_importances.index, legend=False)
plt.title('Ważność Cech w Predykcji Szczęścia (Model XGBoost)')
plt.xlabel('Ważność (Wkład w redukcję błędu)')
plt.ylabel('Czynnik')
plt.show()

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

gbr_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gbr_model.fit(X_train, y_train)
gbr_pred = gbr_model.predict(X_test)

gbr_r2 = r2_score(y_test, gbr_pred)
gbr_rmse = np.sqrt(mean_squared_error(y_test, gbr_pred))

print("\n--- Gradient Boosting Regressor ---")
print("R²:", round(gbr_r2, 4))
print("RMSE:", round(gbr_rmse, 4))

# Walidacja krzyżowa
gbr_cv = cross_val_score(gbr_model, X, y, cv=5, scoring='r2')
print("Cross-val R²:", gbr_cv.mean().round(4))

gbr_feature_importances = pd.Series(gbr_model.feature_importances_, index=X_nor.columns)
gbr_feature_importances = gbr_feature_importances.sort_values(ascending=False)

print("\n--- Ważność Cech w Modelu Gradient Boosting ---")
print(gbr_feature_importances)

plt.figure(figsize=(12, 7))
sns.barplot(x=gbr_feature_importances.values, y=gbr_feature_importances.index, palette="viridis", hue=gbr_feature_importances.index, legend=False)
plt.title('Ważność Cech w Predykcji Szczęścia (Model Gradient Boosting)')
plt.xlabel('Ważność (Wkład w redukcję błędu)')
plt.ylabel('Czynnik')
plt.show()

results = {
    'Linear Regression': {'R2': lr_r2, 'RMSE': lr_rmse},
    'Random Forest': {'R2': rf_r2, 'RMSE': rf_rmse},
    'XGBoost': {'R2': xgb_r2, 'RMSE': xgb_rmse},
    'GradientBoostingRegressor': {'R2': gbr_r2, 'RMSE': gbr_rmse}
}

results_df = pd.DataFrame(results).T.sort_values(by='R2', ascending=False)
print(results_df)

# Wizualizacja porównania R^2 i RMSE
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

results_df['R2'].plot(kind='bar', ax=ax[0], title='Porównanie R^2 Modeli (Im wyżej, tym lepiej)', color=['skyblue', 'lightgreen', 'orange', 'coral'])
ax[0].set_ylabel('R^2 Score')
ax[0].tick_params(axis='x', rotation=45)

results_df['RMSE'].plot(kind='bar', ax=ax[1], title='Porównanie RMSE Modeli (Im niżej, tym lepiej)', color=['skyblue', 'lightgreen', 'orange', 'coral'])
ax[1].set_ylabel('RMSE Score')
ax[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

df['Country'].isna().sum()
df.dropna(subset=['Country'], inplace=True)

happiness_by_country = df.groupby('Country')['Happiness Score'].mean().sort_values(ascending=False)

print("\n--- Top 5 Krajów z Największym Poziomem Szczęścia ---")
print(happiness_by_country.head(5))

plt.figure(figsize=(20, 10))
sns.barplot(x=happiness_by_country.head(15).index, y=happiness_by_country.head(15).values, palette='magma')
plt.title('Top 15 Krajów z Największym Poziomem Szczęścia')
plt.xlabel('Kraj')
plt.ylabel('Średni Poziom Szczęścia')
plt.xticks(rotation=45, ha='right')
for index, value in enumerate(happiness_by_country.head(15).values):
    plt.text(index, value + 0.05, f'{value:.2f}', color='black', ha='center')
plt.show()

print("\n--- Top 5 Krajów z Najniższym Poziomem Szczęścia ---")
print(happiness_by_country.tail(5))

plt.figure(figsize=(20, 10))
sns.barplot(x=happiness_by_country.tail(15).index, y=happiness_by_country.tail(15).values, palette='magma')
plt.title('Top 15 Krajów z Najniższym Poziomem Szczęścia')
plt.xlabel('Kraj')
plt.ylabel('Średni Poziom Szczęścia')
plt.xticks(rotation=45, ha='right')
for index, value in enumerate(happiness_by_country.tail(15).values):
    plt.text(index, value + 0.05, f'{value:.2f}', color='black', ha='center')
plt.show()

import plotly.express as px

fig = px.choropleth(df, 
                    locations='Country', 
                    locationmode='country names',
                    color='Happiness Score',
                    title='World Happiness Score Map')
fig.show()
