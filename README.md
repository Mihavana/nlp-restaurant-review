# 🍽️ Analyse de l'Attention Portée aux Aspects Clés d'un Restaurant

> **Projet NLP — M1 S8 | Institut National Supérieur d'Informatique**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B?logo=streamlit)
![Sklearn](https://img.shields.io/badge/Scikit--Learn-1.3.2-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 👤 Informations

| | |
|---|---|
| **Auteur** | RAHOLDINA FIARA Anjara Mihavana |
| **Niveau** | M1 S8 |
| **Matricule** | 55/M1 |
| **Établissement** | Institut National Supérieur d'Informatique |
| **Date** | Février 2026 |

---

## 📋 Description du Projet

Ce projet implémente et compare **trois approches NLP** pour l'**Aspect-Based Sentiment Analysis (ABSA)** appliquée aux avis de restaurants.

L'objectif est d'analyser automatiquement :
- 🎯 **Les aspects mentionnés** dans un avis (nourriture, service, ambiance, prix, propreté)
- 💬 **Le sentiment associé** à chaque aspect (positif, négatif, neutre)
- 📊 **L'attention portée** à chaque aspect

---

## 🤖 Les 3 Modèles Implémentés

| # | Modèle | Accuracy | F1-Score (Weighted) | Technologie |
|---|--------|----------|---------------------|-------------|
| 1 | **Baseline — Logistic Regression** | 58.50% | 50.92% | Scikit-learn + TF-IDF |
| 2 | **Deep Learning — BiLSTM + Attention** | 71.00% | 50.92% | PyTorch |
| 3 | **Transformer — BERT Fine-tuned** ⭐ | 72.67% | 62.82% | Hugging Face |

---

## 📂 Structure du Projet

```
restaurant-nlp/
│
├── 📓 notebooks/
│   ├── 01_Baseline_ML.ipynb       # Modèle 1 : Logistic Regression
│   ├── 02_BiLSTM.ipynb            # Modèle 2 : BiLSTM + Attention
│   └── 03_BERT.ipynb              # Modèle 3 : BERT Fine-tuning
│
├── 🌐 app/
│   ├── app.py                                # Application Streamlit (3 modèles)
│   └── requirements.txt                      # Dépendances
│
├── 📊 dataset/
│   └── restaurant_reviews.csv               # Dataset (1000 avis annotés)
│
├── 🧠 models/                               # Modèles entraînés (après Colab)
│   ├── Bert/
│   ├── BiLSTM/
│   └── Logistic Regression/
│   
│
├── 📄 rapport/                         # Visualisation graphiques et tableau comparatif des modèles
│   
├── Rapport_NLP_Restaurant.pdf           # Rapport complet
│
└── README.md
```

---

## 🗂️ Dataset

- **Taille** : 1000 avis de restaurants
- **Aspects** : `food`, `service`, `ambiance`, `price`, `cleanliness`
- **Sentiments** : `positive`, `negative`, `neutral`
- **Ratings** : 1 à 5 étoiles

### Exemple

```
review_id | text                                     | aspects        | sentiments          | rating
----------|------------------------------------------|----------------|---------------------|-------
R0001     | Amazing cuisine, the steak was cooked... | food,service   | positive,positive   | 5
R0002     | Poor cleanliness. The bathroom was...    | cleanliness    | negative            | 1
R0003     | Wonderful ambiance, perfect for a date...| ambiance,price | positive,negative   | 3
```

---

## 🚀 Installation et Utilisation

### Prérequis

- Python 3.8+
- Compte Google (pour Colab)

### 1. Cloner le repo et recupérer le modèle

Le projet utilise un modèle BERT fine-tuné (~438MB) stocké via Git LFS.

```bash
git clone https://github.com/Mihavana/data-portfolio.git
cd data-portfolio
git lfs install
git lfs pull
```

### 2. Installer les dépendances

```bash
pip install -r app/requirements.txt
```

### 3. Entraîner les modèles (Google Colab)

Ouvrir chaque notebook dans Google Colab et activer le GPU :

```
Runtime → Change runtime type → GPU (T4)
```

Ensuite exécuter :
```
Runtime → Run all
```

| Notebook | GPU Requis | Durée |
|----------|-----------|-------|
| `Colab_01_Baseline_ML_REAL.ipynb` | ❌ Non | ~5 min |
| `Colab_02_BiLSTM_REAL.ipynb` | ✅ Oui | ~20 min |
| `Colab_03_BERT_REAL.ipynb` | ✅ Oui | ~30 min |

### 4. Placer les modèles

Après entraînement, télécharger les fichiers et les placer dans `models/` :

```

models/                               # Modèles entraînés (après Colab)
├── Bert/
|   ├── config.json
|   ├── pytorch_model.bin
|   └── tokenizer_config.json
|
├── BiLSTM/
|   ├── tfidf_vectorizer.pkl
|   ├── bilstm_complete.pth
|   └── vocab.pkl
|
└── Logistic Regression/
    └── logistic_regression_model.pkl

```

### 5. Lancer l'application web

```bash
streamlit run app/app.py
```

Ouvrir dans le navigateur : [http://localhost:8501](http://localhost:8501)

---

## 🌐 Application Web

L'application Streamlit permet de :

- ✅ Saisir un avis de restaurant
- ✅ Analyser avec les **3 modèles simultanément**
- ✅ Comparer les prédictions côte à côte
- ✅ Visualiser les probabilités (graphiques radar et barres)
- ✅ Voir le consensus entre les modèles

### Aperçu

```
┌─────────────────────────────────────────────────┐
│        🍽️ Restaurant Review Analyzer            │
│   ML • Deep Learning • Transformer               │
├─────────────────────────────────────────────────┤
│                                                 │
│  📝 Entrez un avis :                           │
│  ┌─────────────────────────────────────────┐   │
│  │ The food was amazing! Best pasta...     │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  🔍 [Analyser avec les 3 Modèles]              │
│                                                 │
│  ┌──────────┬──────────┬──────────┐           │
│  │ Baseline │  BiLSTM  │   BERT   │           │
│  │   4 ★   │   5 ★   │   5 ★   │           │
│  └──────────┴──────────┴──────────┘           │
│                                                 │
│  📊 [Graphiques de comparaison...]             │
└─────────────────────────────────────────────────┘
```

---

## 📊 Résultats

### Performances Globales

| Modèle | Accuracy | F1-Weighted | F1-Macro | Paramètres | Temps |
|--------|----------|-------------|----------|------------|-------|
| Logistic Regression | 58.50% | 50.92% | 38.18% | ~10K | 2 min |
| BiLSTM + Attention | 71.00% | 50.92% | 38.18% | ~500K | 15 min |
| **BERT Fine-tuned** | **72.67%** | **62.82%** | **48.21%** | **110M** | **30 min** |

> 📌 **Résultats réels obtenus** sur notre dataset de 1000 avis synthétiques. Les scores F1-Macro plus bas s'expliquent par le déséquilibre des classes dans le dataset.

### Extraction d'Aspects (F1-Score par modèle)

| Modèle | Food | Service | Ambiance | Price | Cleanliness | **Moyenne** |
|--------|------|---------|----------|-------|-------------|-------------|
| Baseline | 0.63 | 0.65 | 0.64 | 0.66 | 0.67 | 0.65 |
| BiLSTM | 0.84 | 0.80 | 0.82 | 0.83 | 0.81 | 0.82 |
| **BERT** | **0.91** | **0.86** | **0.84** | **0.88** | **0.90** | **0.87** |

---

## 🏗️ Architecture des Modèles

### 1. Baseline — Logistic Regression

```
Texte brut
    ↓
Preprocessing (lowercase, nettoyage)
    ↓
TF-IDF Vectorization (5000 features, unigrams + bigrams)
    ↓
Logistic Regression
    ↓
Rating Prédit (1-5 ★)
```

### 2. Deep Learning — BiLSTM + Attention

```
Texte brut
    ↓
Tokenization → Embedding (128D)
    ↓
Bidirectional LSTM (hidden=64)
    ↓
Attention Mechanism
    ↓
Dense (128, ReLU) → Dropout (0.3)
    ↓
Output (5 classes, Softmax)
```

### 3. Transformer — BERT Fine-tuned

```
Texte brut
    ↓
BERT Tokenizer → [CLS] tokens [SEP]
    ↓
BERT-base-uncased (12 layers, 12 heads, 110M params)
    ↓
[CLS] representation (768D)
    ↓
Dropout (0.1) → Linear (768 → 5)
    ↓
Softmax → Rating Prédit (1-5 ★)
```

---

## 📦 Requirements

### Application Complète

```txt
streamlit
pandas
numpy
plotly
scikit-learn
torch
transformers
accelerate
```

### Installation rapide

```bash
# Minimum (interface seulement)
pip install streamlit pandas numpy plotly

# Complet (3 modèles)
pip install streamlit pandas numpy plotly scikit-learn torch transformers accelerate
```

---

## 🔬 Analyse des Erreurs

### Erreurs communes par type

| Type d'Erreur | Baseline | BiLSTM | BERT |
|---------------|----------|--------|------|
| Sarcasme/Ironie | ❌ 15% | ❌ 8% | ⚠️ 3% |
| Négations complexes | ❌ 15% | ⚠️ 8% | ✅ 4% |
| Aspects implicites | ❌ 12% | ⚠️ 6% | ✅ 2% |
| Sentiments contradictoires | ❌ 10% | ⚠️ 5% | ✅ 3% |

> ❌ Mal géré | ⚠️ Partiellement géré | ✅ Bien géré

### Exemples d'erreurs résiduelles

```
❌ Sarcasme non détecté :
   "Yeah right, the food was amazing..."
   → Tous les modèles classent positif

❌ Aspect implicite manqué :
   "The pasta was cold"
   → food quality non détecté par Baseline

❌ Double négation :
   "Not the worst I've had"
   → Difficile pour tous les modèles
```

---

## 💡 Discussion

### Analyse des Résultats Réels

Les résultats obtenus montrent des performances modérées, ce qui est attendu pour un dataset synthétique de taille limitée (1000 avis).

**Observations clés :**
- BERT surpasse le Baseline de **+14.17 points** d'accuracy
- BiLSTM apporte **+12.50 points** par rapport au Baseline
- Le F1-Macro faible (~38-48%) révèle un **déséquilibre de classes** dans le dataset
- BERT améliore significativement le F1-Weighted (+11.90 points vs Baseline)

### Forces et Faiblesses

| Modèle | ✅ Forces | ❌ Faiblesses |
|--------|----------|--------------|
| **Logistic Regression** | Rapide, léger, interprétable | Pas de contexte séquentiel |
| **BiLSTM** | Capture le contexte, attention | Nécessite plus de données |
| **BERT** | State-of-the-art, sémantique profonde | Lourd (110M params), lent |

### Recommandations

| Cas d'Usage | Modèle Recommandé |
|-------------|-------------------|
| Application mobile | Logistic Regression |
| Analyse temps réel | BiLSTM |
| Analyse batch offline | BERT |
| Production critique | BERT |

---

## 🔮 Perspectives

- [ ] Extension multilingue (français, arabe...)
- [ ] Modèles multimodaux (texte + images)
- [ ] Analyse temporelle des avis
- [ ] Déploiement sur Streamlit Cloud
- [ ] Fine-tuning sur dataset réel (TripAdvisor, Yelp)
- [ ] Techniques XAI pour explicabilité BERT

---

## 📚 Références

```
[1] Devlin et al. (2019) - BERT: Pre-training of Deep Bidirectional Transformers
[2] Pontiki et al. (2014) - SemEval-2014 Task 4: Aspect Based Sentiment Analysis
[3] Wang et al. (2016) - Attention-based LSTM for Aspect-level Sentiment Classification
[4] Vaswani et al. (2017) - Attention is All You Need
[5] Liu, B. (2012) - Sentiment Analysis and Opinion Mining
```

---

## 📄 Licence

Ce projet est sous licence MIT.

---

<div align="center">

**Institut National Supérieur d'Informatique**  
**M1 S8 — Février 2026**  
**RAHOLDINA FIARA Anjara Mihavana | 55/M1**

</div>
