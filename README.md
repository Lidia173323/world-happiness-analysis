# 🌍 World Happiness Report - Analiza i Predykcja

![World Happiness Map](world_happiness_map.png)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FqkGSzzhdrPNSACvOPQC2y-Fbg0dUxoa?usp=sharing)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-green.svg)](https://pandas.pydata.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Kompleksowa analiza danych World Happiness Report z wykorzystaniem technik uczenia maszynowego do predykcji poziomu szczęścia krajów na podstawie czynników ekonomicznych i społecznych.

## 📊 O Projekcie

Projekt analizuje dane z World Happiness Report (lata 2015-2016) w celu zidentyfikowania kluczowych czynników wpływających na poziom szczęścia w różnych krajach świata. Wykorzystuje zaawansowane modele uczenia maszynowego do predykcji wskaźnika szczęścia.

### 🎯 Główne Cele

- **Eksploracyjna Analiza Danych (EDA)** - Identyfikacja wzorców i trendów
- **Analiza Korelacji** - Badanie zależności między czynnikami szczęścia
- **Modelowanie Predykcyjne** - Prognozowanie poziomu szczęścia
- **Wizualizacja Wyników** - Interaktywne mapy i wykresy

## 🔍 Analizowane Czynniki

| Czynnik | Opis |
|---------|------|
| **Economy (GDP per Capita)** | PKB per capita - wskaźnik ekonomiczny |
| **Family** | Wsparcie społeczne i rodzinne |
| **Health (Life Expectancy)** | Oczekiwana długość życia |
| **Freedom** | Wolność podejmowania decyzji życiowych |
| **Trust (Government Corruption)** | Zaufanie do rządu, percepcja korupcji |
| **Dystopia Residual** | Reszta dystopii (wartość bazowa) |

## 🛠️ Technologie i Biblioteki

### Analiza Danych
```python
pandas
numpy
```

### Wizualizacja
```python
matplotlib
seaborn
plotly
```

### Machine Learning
```python
scikit-learn
xgboost
```

## 📦 Instalacja

### Krok 1: Sklonuj repozytorium
```bash
git clone https://github.com/Lidia173323/world-happiness-analysis.git
cd world-happiness-analysis
```

### Krok 2: Stwórz wirtualne środowisko (opcjonalnie)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# lub
venv\Scripts\activate  # Windows
```

### Krok 3: Zainstaluj zależności
```bash
pip install -r requirements.txt
```

### Krok 4: Uruchom analizę
```bash
python world_happiness_analysis.py
```

## 📁 Struktura Projektu

```
world-happiness-analysis/
│
├── README.md                      # Dokumentacja projektu
├── requirements.txt               # Zależności Python
├── .gitignore                     # Pliki ignorowane przez Git
├── world_happiness_analysis.py   # Główny skrypt analizy
└── world_happiness_report.csv    # Dane źródłowe (2015-2016)
```

## 🔬 Metodologia

### 1. Preprocessing Danych
- Wczytanie i inspekcja danych
- Analiza brakujących wartości
- Filtracja danych (lata 2015-2016)
- Usunięcie duplikatów

### 2. Eksploracyjna Analiza Danych (EDA)
- Statystyki opisowe
- Rozkłady zmiennych
- Analiza korelacji
- Wizualizacje (histogramy, boxploty, pairplots)

### 3. Przygotowanie Danych do Modelowania
- Standaryzacja cech (StandardScaler)
- Podział na zbiór treningowy (80%) i testowy (20%)
- Walidacja krzyżowa (5-fold CV)

### 4. Modele Uczenia Maszynowego

#### 🤖 Implementowane Modele

| Model | Typ | Zastosowanie |
|-------|-----|--------------|
| **Linear Regression** | Podstawowy | Model bazowy, interpretacja zależności |
| **Random Forest** | Ensemble | Redukcja przeuczenia, feature importance |
| **XGBoost** | Gradient Boosting | Wysoka dokładność, optymalizacja |
| **Gradient Boosting** | Ensemble | Sekwencyjne uczenie, redukcja błędu |

## 📈 Wyniki Modeli

### Metryki Oceny
- **R² (Coefficient of Determination)** - Jak dobrze model wyjaśnia wariancję
- **RMSE (Root Mean Squared Error)** - Średni błąd predykcji
- **Cross-validation R²** - Stabilność modelu na różnych podzbiorach

### Przykładowe Wyniki

```
Model                        R²      RMSE    CV R²
────────────────────────────────────────────────
XGBoost                    0.9845   0.1234  0.9801
Random Forest              0.9812   0.1356  0.9778
Gradient Boosting          0.9790   0.1432  0.9765
Linear Regression          0.7856   0.4589  0.7723
```

## 🎨 Wizualizacje

### 1. Mapa Szczęścia Świata
Interaktywna mapa choropleth pokazująca rozkład poziomu szczęścia w różnych krajach.

### 2. Analiza Ważności Cech
Wykresy pokazujące, które czynniki mają największy wpływ na poziom szczęścia według modeli ML.

### 3. Korelacje Między Zmiennymi
Heatmapy pokazujące wzajemne zależności między czynnikami szczęścia.

### 4. Top Kraje
- **Najszczęśliwsze kraje** - Top 15 krajów z najwyższym wskaźnikiem
- **Najmniej szczęśliwe kraje** - Top 15 krajów z najniższym wskaźnikiem

## 🔑 Kluczowe Odkrycia

1. **Ekonomia ma znaczenie** - GDP per capita jest silnie skorelowane ze szczęściem
2. **Zdrowie i długość życia** - Life Expectancy jest kluczowym predyktorem
3. **Wsparcie społeczne** - Family/Social Support znacząco wpływa na szczęście
4. **Wolność wyboru** - Freedom ma większe znaczenie niż się powszechnie sądzi
5. **Modele ensemble dominują** - XGBoost i Random Forest osiągają najlepsze wyniki

## 🚀 Przyszłe Usprawnienia

- [ ] Dodanie danych z kolejnych lat (2017-2024)
- [ ] Implementacja deep learning (Neural Networks)
- [ ] Analiza szeregów czasowych i trendów
- [ ] Dashboard interaktywny (Streamlit/Dash)
- [ ] Analiza klastrów krajów o podobnym profilu szczęścia
- [ ] Predykcja przyszłych trendów
- [ ] API do predykcji poziomu szczęścia

## 📚 Źródła Danych

- [World Happiness Report](https://worldhappiness.report/)
- [Kaggle - World Happiness Dataset](https://www.kaggle.com/)

## 🤝 Jak Przyczynić się do Projektu

1. Fork projektu
2. Stwórz branch dla swojej funkcji (`git checkout -b feature/AmazingFeature`)
3. Commit zmian (`git commit -m 'Add some AmazingFeature'`)
4. Push do brancha (`git push origin feature/AmazingFeature`)
5. Otwórz Pull Request

## 📝 Licencja

Projekt jest dostępny na licencji MIT. Zobacz plik `LICENSE` dla szczegółów.

## 👨‍💻 Autor

**Lidia Furgał**
- GitHub: [@Lidia173323](https://github.com/Lidia173323)
- LinkedIn: [Lidia Furgał](https://www.linkedin.com/in/lidiafurgal/)

## 🙏 Podziękowania

- World Happiness Report Team za udostępnienie danych
- Społeczność open-source za nieocenione narzędzia
- Wszyscy contributors projektu

---

⭐ Jeśli projekt Ci się podoba, zostaw gwiazdkę na GitHubie!

**Ostatnia aktualizacja:** Grudzień 2025
