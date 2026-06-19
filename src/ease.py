import argparse
import os

import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm
from utils import *

# ──────────────────────────────────────────────────────────────────────────────
# EASE: Embarrassingly Shallow Autoencoders for Sparse Data
# Paper: https://arxiv.org/abs/1905.03375
#
# Formula:
#   G = X^T X + lambda * I          (item x item gram matrix)
#   B = G^{-1}                      (inverse)
#   W_ij = -B_ij / B_jj             (element-wise column division)
#   diag(W) = 0                     (no self-recommendation)
#   score(u) = X[u] @ W
#
# Why EASE suits this dataset:
#   - Closed-form solution, no iteration → no overfitting via gradient descent
#   - Extremely robust on sparse data (purchase density ≈ 0%)
#   - Item count 29,502 → W is ~3.5 GB float32, feasible on most servers
# ──────────────────────────────────────────────────────────────────────────────

TOPK = 10


def train_ease(X: sparse.csr_matrix, lambda_: float) -> np.ndarray:
    """
    Returns W (n_items x n_items, float32).
    Uses float64 internally to avoid numerical precision issues.
    Memory: ~3.5 GB for 29,502 items in float32.
    """
    n_items = X.shape[1]

    print(f"  Computing X^T X ({n_items} x {n_items}) ...")
    G = (X.T @ X).toarray().astype(np.float64)

    print(f"  Adding regularization (lambda={lambda_}) to diagonal ...")
    np.fill_diagonal(G, G.diagonal() + lambda_)

    print("  Computing matrix inverse ...")
    B = np.linalg.inv(G)
    del G

    print("  Building weight matrix W ...")
    diag_B = np.diag(B)                        # (n_items,)
    W = -B / diag_B[np.newaxis, :]             # divide each col j by B_jj
    np.fill_diagonal(W, 0.0)
    del B

    return W.astype(np.float32)


def topk_indices(scores_row: np.ndarray, k: int) -> np.ndarray:
    """
    Returns indices of top-k elements in descending order.
    Handles -inf and NaN safely.
    """
    # Replace NaN with -inf so they rank last
    scores_row = np.where(np.isnan(scores_row), -np.inf, scores_row)
    # argpartition is O(n), faster than full argsort
    if len(scores_row) <= k:
        idx = np.argsort(scores_row)[::-1]
    else:
        part = np.argpartition(scores_row, -k)[-k:]
        idx  = part[np.argsort(scores_row[part])[::-1]]
    return idx


def pad_with_popular(rec_list: list, popular_items: list, k: int) -> list:
    """
    Pads rec_list up to length k using popular_items,
    skipping any items already in rec_list.
    """
    if len(rec_list) >= k:
        return rec_list[:k]
    item_set = set(rec_list)
    for pop in popular_items:
        if pop not in item_set:
            rec_list.append(pop)
            item_set.add(pop)
        if len(rec_list) == k:
            break
    return rec_list


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",   default="train.parquet", type=str)
    parser.add_argument("--dir_path",   default="../data/",      type=str)
    parser.add_argument("--output_dir", default="../output/",    type=str)

    # EASE
    parser.add_argument(
        "--lambda_", type=float, default=500.0,
        help="Regularization. Try 100, 500, 1000. Larger = stronger regularization."
    )

    # Event weights (same as train_als.py)
    parser.add_argument("--w_view",     type=float, default=1.0)
    parser.add_argument("--w_cart",     type=float, default=100.0)
    parser.add_argument("--w_purchase", type=float, default=500.0)

    # Time decay (same as train_als.py)
    parser.add_argument(
        "--time_decay_lambda", type=float, default=0.03,
        help="Time decay: confidence *= exp(-lambda * days_ago)."
    )

    # Inference
    parser.add_argument(
        "--chunk_size", type=int, default=5000,
        help="Users processed per chunk. Reduce if out-of-memory."
    )
    parser.add_argument(
        "--filter_interacted", action="store_true", default=True,
        help="Exclude all interacted items from recommendations."
    )

    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Load
    print("Loading data ...")
    train_df = pd.read_parquet(os.path.join(args.dir_path, args.data_dir))

    user2idx = {v: k for k, v in enumerate(train_df['user_id'].unique())}
    idx2user = {k: v for k, v in enumerate(train_df['user_id'].unique())}
    item2idx = {v: k for k, v in enumerate(train_df['item_id'].unique())}
    idx2item = {k: v for k, v in enumerate(train_df['item_id'].unique())}

    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"  Users: {n_users:,}  |  Items: {n_items:,}")

    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)

    # 2. Event weights
    event_weights = {
        'view':     args.w_view,
        'cart':     args.w_cart,
        'purchase': args.w_purchase,
    }
    train_df["label"] = train_df["event_type"].map(event_weights).fillna(1.0)
    print(f"  Event weights: view={args.w_view}  cart={args.w_cart}  purchase={args.w_purchase}")

    # 3. Time decay
    train_df['event_time'] = pd.to_datetime(train_df['event_time'])
    max_date = train_df['event_time'].max()
    days_ago = (max_date - train_df['event_time']).dt.days
    train_df["label"] = train_df["label"] * np.exp(-args.time_decay_lambda * days_ago)
    print(f"  Time decay lambda: {args.time_decay_lambda}")

    # 4. Sparse matrix
    print("Building sparse matrix ...")
    user_item = train_df.groupby(["user_idx", "item_idx"])["label"].sum().reset_index()
    X = sparse.csr_matrix(
        (user_item["label"].values,
         (user_item["user_idx"].values,
          user_item["item_idx"].values)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    print(f"  Shape: {X.shape}  |  nnz: {X.nnz:,}")
    mem_gb = (n_items ** 2 * 4) / (1024 ** 3)
    print(f"  Estimated W matrix memory: {mem_gb:.1f} GB")

    # 5. Popular items (fallback padding)
    popular_items = (train_df[train_df['event_type'] == 'purchase']
                     .groupby('item_idx').size()
                     .sort_values(ascending=False)
                     .index.tolist())
    if len(popular_items) == 0:
        popular_items = (train_df.groupby('item_idx').size()
                         .sort_values(ascending=False).index.tolist())
    print(f"  Popular item pool: {len(popular_items):,}")

    # 6. Train EASE
    print(f"\nTraining EASE (lambda={args.lambda_}) ...")
    W = train_ease(X, lambda_=args.lambda_)
    print(f"  W shape: {W.shape}  dtype: {W.dtype}")

    # 7. Chunked inference
    print(f"\nRunning inference (chunk_size={args.chunk_size}) ...")
    all_users = np.arange(n_users)
    results   = []
    n_padded  = 0

    for start in tqdm(range(0, n_users, args.chunk_size)):
        end     = min(start + args.chunk_size, n_users)
        u_chunk = all_users[start:end]

        X_chunk = X[u_chunk].toarray()       # (chunk, n_items) float32
        scores  = X_chunk @ W                # (chunk, n_items) float32

        # Filter all interacted items (set score to -inf)
        if args.filter_interacted:
            scores[X_chunk > 0] = -np.inf

        for i, uid in enumerate(u_chunk):
            top = topk_indices(scores[i], TOPK)   # indices sorted desc
            items = top.tolist()

            # Pad if fewer than TOPK valid (non -inf) items were found
            valid = [it for it in items if not np.isinf(scores[i, it])]
            if len(valid) < TOPK:
                valid = pad_with_popular(valid, popular_items, TOPK)
                n_padded += 1
                items = valid
            else:
                items = items[:TOPK]

            for iid in items:
                results.append((uid, iid))

    print(f"  Users padded with popular items: {n_padded:,}")

    # 8. Build & validate submission
    print("\nBuilding submission ...")
    sub_df = pd.DataFrame(results, columns=['user_id', 'item_id'])
    sub_df['user_id'] = sub_df['user_id'].map(idx2user)
    sub_df['item_id'] = sub_df['item_id'].map(idx2item)

    expected_rows = n_users * TOPK
    actual_rows   = len(sub_df)

    # Validation (hard stop if wrong)
    assert actual_rows == expected_rows, \
        f"Row count mismatch: {actual_rows:,} != {expected_rows:,}"
    assert sub_df['user_id'].nunique() == n_users, \
        "User count mismatch"
    assert sub_df.groupby('user_id').size().eq(TOPK).all(), \
        "Some users do not have exactly 10 recommendations"

    # Save 
    outdir = args.output_dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    out_path = os.path.join(outdir, "output_ease.csv")
    sub_df.to_csv(out_path, index=False)

    print(f"\nSaved → {out_path}")
    print(f"  Total rows  : {actual_rows:,}  ✓")
    print(f"  Unique users: {sub_df['user_id'].nunique():,}  ✓")
    print(f"  Padded users: {n_padded:,}")
    print(f"  lambda_     : {args.lambda_}")
    print(f"  time_decay  : {args.time_decay_lambda}")


if __name__ == "__main__":
    main()