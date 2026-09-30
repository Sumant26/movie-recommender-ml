# 🎬 Movie Recommender

Collaborative-filtering recommender on MovieLens: item-item cosine similarity, then matrix factorisation with PyTorch embeddings, plus personalised recommendations for you.

**Skills:** collaborative filtering · sparse matrices · cosine similarity · embeddings · matrix factorisation · regularisation · RMSE · hit rate @ 10 · cold start

**Time:** one to two weeks · **Level:** intermediate

## Run it
1. Open [Google Colab](https://colab.research.google.com) → **File → Upload notebook** → `movie_recommender.ipynb`.
2. Optional: **Runtime → Change runtime type → T4 GPU** (it also runs fine on CPU).
3. Run top to bottom. Edit `my_ratings` in section 9 to get your own recommendations.

## Expected results (MovieLens small)
| Model | Test RMSE (stars) |
|---|---|
| Global average | ~1.04 |
| User + movie bias | ~0.87 |
| Matrix factorisation (k=32, λ=0.1) | ~0.84 |

| Ranking (hit rate @ 10 among 100 candidates) | |
|---|---|
| Random | 10% |
| Most popular | ~76% |
| Matrix factorisation alone | ~52% |
| MF + popularity blend | ~80% |

Data: [MovieLens](https://grouplens.org/datasets/movielens/) by GroupLens Research.
# movie-recommender-ml
