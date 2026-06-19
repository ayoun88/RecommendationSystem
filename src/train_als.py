import argparse
import os

import pandas as pd
import numpy as np
from scipy import sparse
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
from utils import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="train.parquet", type=str)
    parser.add_argument("--dir_path", default="../data/", type=str)
    parser.add_argument("--output_dir", default="../output/", type=str)

    # model args
    parser.add_argument("--num_factor", help="The number of latent factors to compute", type=int,default=512)
    parser.add_argument(
        "--regularization", type=int, default=0.001, help="The regularization factor to use"
    )
    parser.add_argument(
        "--alpha", type=int, default=10, help="governs the baseline confidence in preference observations"
    )
    parser.add_argument(
        "--time_decay_lambda", type=float, default=0.03,
    )

    # train args
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    train_df = pd.read_parquet(os.path.join(args.dir_path, args.data_dir))

    user2idx = {v: k for k, v in enumerate(train_df['user_id'].unique())}
    idx2user = {k: v for k, v in enumerate(train_df['user_id'].unique())}
    item2idx = {v: k for k, v in enumerate(train_df['item_id'].unique())}
    idx2item = {k: v for k, v in enumerate(train_df['item_id'].unique())}

    # Apply the mapping functions to 'user_id' and 'item_id' columns
    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)

    # [CHANGE 1] Event type weights
    # Original: all events = 1 (view, cart, purchase treated equally)
    # Updated : purchase(500) >> cart(100) >> view(1)
    # Reason  : view:cart:purchase ratio = 4015:8:1 in this dataset
    #           purchase/cart signals are extremely rare and must be amplified
    event_weights = {'view': 1, 'cart': 0, 'purchase': 500}
    train_df["label"] = train_df["event_type"].map(event_weights).fillna(1)

    # [CHANGE 2] Time decay — recent events weighted higher
    # Reason: week-9 (late Feb) purchase spike means recent data dominates
    #         exp(-lambda * days_ago): days_ago=0 -> weight=1.0, days_ago=120 -> weight~0.03 (lambda=0.03)
    train_df['event_time'] = pd.to_datetime(train_df['event_time'])
    max_date = train_df['event_time'].max()
    days_ago = (max_date - train_df['event_time']).dt.days
    train_df["label"] = train_df["label"] * np.exp(-args.time_decay_lambda * days_ago)

    user_item_matrix = train_df.groupby(["user_idx", "item_idx"])["label"].sum().reset_index()

    sparse_user_item = sparse.csr_matrix(
                                        (user_item_matrix["label"].values,
                                        (user_item_matrix["user_idx"].values,
                                        user_item_matrix["item_idx"].values)),
                                        shape=(len(user2idx), len(item2idx)),
                                        dtype=np.float32)
    sparse_user_item = sparse_user_item.tocsr()
    # ref: https://github.com/benfred/implicit/blob/main/examples/movielens.py
    num_factor = args.num_factor
    regularization = args.regularization
    alpha = args.alpha

    model = AlternatingLeastSquares(
        factors=num_factor,
        regularization=regularization,
        alpha=alpha,
        use_gpu=True)
    
    model.fit(sparse_user_item)

    test_users_idx = np.array(train_df['user_idx'].unique())
    test_users_idx_li = [num for num in test_users_idx for _ in range(10)]
    # [CHANGE 3] filter_already_liked_items=True
    # Original: False (already-purchased items can be re-recommended)
    # Updated : True  (exclude already-purchased items from recommendations)
    # Reason  : repeat purchase rate is only 7% in this dataset -> re-recommending wastes slots
    public_outputs = model.recommend(test_users_idx, sparse_user_item[test_users_idx], N=10, filter_already_liked_items=False)

    recommend_items = public_outputs[0]
    sub_df = pd.DataFrame({'user_id' : test_users_idx_li, 'item_id' : recommend_items.flatten()})
    sub_df['user_id'] = sub_df['user_id'].map(idx2user)
    sub_df['item_id'] = sub_df['item_id'].map(idx2item)

    
    outdir = args.output_dir
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    sub_df.to_csv(os.path.join(outdir,"output_als.csv"), index=False)

if __name__ == "__main__":
    main()