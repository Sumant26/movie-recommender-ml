# 🎬 Movie Recommender System (Collaborative Filtering & PyTorch Matrix Factorization)

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning recommender system built on the **MovieLens** dataset. It demonstrates how modern recommendation engines (like Netflix, Spotify, Amazon, and YouTube) predict user preferences and generate personalized recommendations using **Collaborative Filtering**, **Cosine Similarity on Sparse Matrices**, and **Deep Matrix Factorization with PyTorch Embeddings**.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Algorithms & Architecture](#-algorithms--architecture)
  - [1. Baseline Models (Global Mean & User/Item Bias)](#1-baseline-models)
  - [2. Item-Item Collaborative Filtering (Cosine Similarity)](#2-item-item-collaborative-filtering)
  - [3. Neural Matrix Factorization (PyTorch Embeddings)](#3-neural-matrix-factorization)
  - [4. Ranking Evaluation (Hit Rate @ 10 & Popularity Blending)](#4-ranking-evaluation)
  - [5. Cold-Start Personalization](#5-cold-start-personalization)
- [Benchmark Results](#-benchmark-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Option 1: Google Colab (Recommended)](#option-1-google-colab-recommended)
  - [Option 2: Local Setup](#option-2-local-setup)
- [How to Personalize Recommendations](#-how-to-personalize-recommendations)
- [Future Extensions](#-future-extensions)
- [Dataset & Acknowledgments](#-dataset--acknowledgments)

---

## 📖 Overview

Recommender systems solve two fundamental challenges:
1. **Extreme Sparsity**: In a typical user-item interaction matrix ($610 \text{ users} \times 9,724 \text{ movies} \approx 5.9\text{M potential pairs}$), less than **1.7%** of entries are known. The goal is to accurately predict the remaining **98.3%**.
2. **The Long Tail**: A few popular blockbusters receive the vast majority of ratings, while thousands of niche titles receive very few.

This project walks through progressively sophisticated models: from simple statistical baselines to item-item memory-based filtering and latent embedding matrix factorization, evaluating both **rating prediction accuracy (RMSE)** and **top-$N$ ranking quality (Hit Rate @ 10)**.

---

## ✨ Key Features

- **Automated Dataset Pipeline**: Automatically downloads and extracts the [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) dataset (~100k ratings across 9,700+ movies).
- **Exploratory Data Analysis (EDA)**: Visualizations for rating distribution, sparsity matrix density, and long-tail power law analysis.
- **Sparse Matrix Item-Based CF**: High-performance cosine similarity on mean-centered ratings using `scipy.sparse.csr_matrix`.
- **PyTorch Matrix Factorization Engine**: Custom `nn.Module` learning low-dimensional latent embeddings for users and movies with regularized bias terms.
- **Latent Space Visualization**: Dimensionality reduction using **PCA** on learned movie embeddings to visually inspect clusters of similar movie genres/themes.
- **Top-$N$ Hit Rate Ranking Benchmark**: Evaluates model performance using a sampled candidate ranking protocol (1 target positive mixed with 99 unrated distractors).
- **Hybrid Scoring**: Blends collaborative filtering prediction scores with log-popularity signals for state-of-the-art top-$N$ recommendation accuracy.
- **Instant Cold-Start Inference**: Allows a new user to enter a few movie ratings and dynamically learns their user embedding vector with frozen movie weights.

---

## 🧠 Algorithms & Architecture

### 1. Baseline Models
- **Global Average**: $\hat{r}_{u,i} = \mu$
- **Regularized Bias Model**:
  $$\hat{r}_{u,i} = \mu + b_u + b_i$$
  where user bias $b_u$ and movie bias $b_i$ are regularized toward zero for entities with few ratings using shrinkage parameter $\lambda_{reg}$:
  $$b_i = \frac{\sum_{u \in R_i} (r_{u,i} - \mu)}{|R_i| + \lambda_{reg}}, \quad b_u = \frac{\sum_{i \in R_u} (r_{u,i} - \mu - b_i)}{|R_u| + \lambda_{reg}}$$

### 2. Item-Item Collaborative Filtering
- Mean-centered user ratings: $s_{u,i} = r_{u,i} - \bar{r}_u$.
- Movie-to-movie similarity computed via Cosine Similarity over sparse vectors:
  $$\text{sim}(i, j) = \frac{\mathbf{s}_i \cdot \mathbf{s}_j}{\|\mathbf{s}_i\|_2 \|\mathbf{s}_j\|_2}$$

### 3. Neural Matrix Factorization
Learns $k$-dimensional latent representations $\mathbf{p}_u \in \mathbb{R}^k$ and $\mathbf{q}_i \in \mathbb{R}^k$ for each user $u$ and movie $i$:

$$\hat{r}_{u,i} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i$$

```
         Movies (N)                             Latent Factors (k)        Movies (N)
Users (M) [ ?  4.0  ?  5.0 ... ]   ≈   Users (M) [ user vectors ] × k [ movie vectors ]
```

**Optimization Objective (Loss with $L_2$ Regularization):**
$$\mathcal{L} = \frac{1}{|R|} \sum_{(u, i) \in R} \left(\hat{r}_{u,i} - r_{u,i}\right)^2 + \lambda \left( \|\mathbf{p}_u\|_2^2 + \|\mathbf{q}_i\|_2^2 + b_u^2 + b_i^2 \right)$$

Trained via mini-batch stochastic gradient descent with the **Adam** optimizer (`lr=0.005`, `batch_size=1024`, `k=32`, `lambda=0.1`).

### 4. Ranking Evaluation
- **Hit Rate @ 10 Protocol**: For users in the test set who rated a movie $\ge 4.5$ stars, mix that movie with 99 random unrated candidates. Rank all 100 items; a "hit" occurs if the ground-truth favorite is in the top 10.
- **Popularity-MF Hybrid Blend**:
  $$S_{\text{hybrid}}(u, i) = (1 - w) \cdot Z(S_{\text{MF}}(u, i)) + w \cdot Z(\log(1 + \text{count}_i))$$
  where $Z(\cdot)$ standardizes scores ($z$-score normalization) and $w \in [0, 1]$ balances personalized taste and item popularity.

### 5. Cold-Start Personalization
When a new user provides ratings:
1. Movie embeddings $\mathbf{q}_i$ and biases $b_i$ are kept **frozen**.
2. A lightweight optimization solves for only $\mathbf{p}_{\text{new}}$ and $b_{\text{new}}$ over ~300 iterations ($\approx 50\text{ ms}$).
3. Top-$N$ recommendations are immediately scored across the catalog via $\mathbf{Q} \mathbf{p}_{\text{new}} + \mathbf{b}_{\text{movie}} + b_{\text{new}} + \mu$.

---

## 📊 Benchmark Results

### 1. Rating Prediction Error (MovieLens 100k)
| Model | Test RMSE (Stars) | Description |
|---|:---:|---|
| **Global Average** | ~1.040 | Fixed baseline predicting catalog mean |
| **User + Movie Bias** | ~0.871 | Regularized entity offsets ($\lambda=5$) |
| **Matrix Factorization ($k=32, \lambda=0.1$)** | **~0.840** | Latent factor dot-product with embeddings |

### 2. Top-10 Ranking Performance (Hit Rate @ 10 among 100 candidates)
| Ranking Strategy | Hit Rate @ 10 | Key Takeaway |
|---|:---:|---|
| **Random Guess** | 10.0% | Theoretical lower bound ($10 / 100$) |
| **Matrix Factorization (Alone)** | ~52.0% | Strong personalization on rated items |
| **Most Popular Baseline** | ~76.0% | Strong baseline for candidate discovery |
| **MF + Popularity Blend ($w=0.5$)** | **~80.0%+** | **Best overall ranking performance** |

---

## 📂 Project Structure

```
06-movie-recommender/
├── README.md               # Project documentation and benchmarks
├── movie_recommender.ipynb # Complete Jupyter Notebook (EDA, Models, PyTorch MF, Evaluation)
└── movie_mf.pt             # (Generated) PyTorch checkpoint with trained weights and ID mappings
```

---

## 🚀 Getting Started

### Option 1: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com).
2. Click **File → Upload notebook** and upload [`movie_recommender.ipynb`](movie_recommender.ipynb).
3. (Optional for acceleration): Navigate to **Runtime → Change runtime type → T4 GPU → Save**.
4. Run all cells sequentially (**Runtime → Run all** or `Shift + Enter`).

### Option 2: Local Setup

#### 1. Clone the repository
```bash
git clone https://github.com/Sumant26/movie-recommender-ml.git
cd movie-recommender-ml
```

#### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu # or cu118 / cu121 for GPU
pip install numpy pandas matplotlib scipy scikit-learn jupyter
```

#### 4. Launch Jupyter Notebook
```bash
jupyter notebook movie_recommender.ipynb
```

---

## 🍿 How to Personalize Recommendations

In **Section 9** of the notebook, supply your own favorite and disliked movies in the `my_ratings` dictionary:

```python
my_ratings = {
    "Toy Story (1995)": 4.5,
    "Matrix, The (1999)": 5.0,
    "Titanic (1997)": 2.0,
    "Lord of the Rings: The Fellowship of the Ring, The (2001)": 5.0,
    "Notebook, The (2004)": 1.5,
    "Inception (2010)": 5.0,
}
```

> **Tip:** Use the helper function `find_movie("Inception")` to look up exact title strings from the catalog.

Run the cell to instantly optimize your taste vector and view your tailored top 15 recommendations.

---

## 🔮 Future Extensions

- **Content-Based & Hybrid Features**: Vectorize movie genres, tags, and synopses with TF-IDF or text transformers (e.g., MiniLM) to combine metadata with collaborative signals.
- **Implicit Feedback / Neural Collaborative Filtering (NCF)**: Upgrade from explicit ratings (MSE loss) to implicit pairwise ranking loss (e.g., Bayesian Personalized Ranking - BPR loss).
- **Interactive UI**: Deploy a web demo using **Streamlit** or **Gradio** allowing users to pick movies from dropdowns and receive instant recommendations.
- **Scale Up**: Test and scale the PyTorch model pipeline to larger datasets like **MovieLens 1M**, **10M**, or **25M**.

---

## 📚 Dataset & Acknowledgments

- **Dataset**: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19. [GroupLens Research](https://grouplens.org/datasets/movielens/).
- Built with [PyTorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/), and [pandas](https://pandas.pydata.org/).
