"""
ensemble.py

ALS + SASRec 결과를 앙상블해서 최종 추천 생성.

사용법:
    python ensemble.py --als_file ../output/output_als.csv \
                       --sasrec_file ../output/output_sasrec.csv \
                       --als_weight 0.7 \
                       --sasrec_weight 0.3

앙상블 방식: Weighted Reciprocal Rank Fusion (RRF)
    - 각 모델의 순위(rank)를 점수로 변환 후 가중합
    - rank 1 = 1.0점, rank 10 = 0.1점 (선형)
    - score 정규화 불필요 → 구현 단순하고 안정적
"""

import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


TOPK = 10


def rank_to_score(n: int) -> np.ndarray:
    """
    rank 1~n을 1.0~0.1 선형 점수로 변환.
    rank 1 (최상위) = 1.0
    rank n (최하위) = 1/n
    """
    return np.array([(n - r) / n for r in range(n)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--als_file",      default="../output/output_als.csv",    type=str)
    parser.add_argument("--sasrec_file",   default="../output/output_sasrec.csv", type=str)
    parser.add_argument("--output_dir",    default="../output/",                  type=str)
    parser.add_argument("--als_weight",    default=0.7, type=float,
        help="ALS 가중치. als_weight + sasrec_weight = 1.0 권장")
    parser.add_argument("--sasrec_weight", default=0.3, type=float,
        help="SASRec 가중치")
    args = parser.parse_args()

    print(f"ALS weight: {args.als_weight}  |  SASRec weight: {args.sasrec_weight}")

    # ── 1. 파일 로드 ──────────────────────────────────────────────────────
    print("\nLoading files ...")
    als    = pd.read_csv(args.als_file)
    sasrec = pd.read_csv(args.sasrec_file)

    print(f"  ALS    : {len(als):,}행  |  {als['user_id'].nunique():,}명")
    print(f"  SASRec : {len(sasrec):,}행  |  {sasrec['user_id'].nunique():,}명")

    # ── 2. rank score 계산 ────────────────────────────────────────────────
    # 각 모델의 추천 순서를 점수로 변환
    # groupby cumcount: 같은 유저 내에서 0, 1, 2, ... 순서 부여
    als['rank']    = als.groupby('user_id').cumcount()       # 0~9
    sasrec['rank'] = sasrec.groupby('user_id').cumcount()    # 0~9

    # rank → score (rank 0 = 1.0, rank 9 = 0.1)
    als['als_score']       = (TOPK - als['rank'])    / TOPK * args.als_weight
    sasrec['sasrec_score'] = (TOPK - sasrec['rank']) / TOPK * args.sasrec_weight

    # ── 3. 두 결과 합산 ───────────────────────────────────────────────────
    print("Merging scores ...")
    als_slim    = als[['user_id', 'item_id', 'als_score']]
    sasrec_slim = sasrec[['user_id', 'item_id', 'sasrec_score']]

    # outer join: 한 모델에만 있는 아이템도 포함
    merged = pd.merge(als_slim, sasrec_slim,
                      on=['user_id', 'item_id'], how='outer')
    merged['als_score']    = merged['als_score'].fillna(0)
    merged['sasrec_score'] = merged['sasrec_score'].fillna(0)
    merged['final_score']  = merged['als_score'] + merged['sasrec_score']

    # ── 4. 유저별 top 10 추출 ─────────────────────────────────────────────
    print("Selecting top 10 per user ...")
    result = (merged
              .sort_values(['user_id', 'final_score'], ascending=[True, False])
              .groupby('user_id')
              .head(TOPK)
              .reset_index(drop=True)
              [['user_id', 'item_id']])

    # ── 5. 혹시 10개 미만인 유저 확인 & 보완 ─────────────────────────────
    user_counts = result.groupby('user_id').size()
    short_users = user_counts[user_counts < TOPK]

    if len(short_users) > 0:
        print(f"  WARNING: {len(short_users):,}명이 {TOPK}개 미만 → ALS 결과로 보완")
        # ALS 결과에서 부족한 슬롯 채우기
        als_dict = als.groupby('user_id')['item_id'].apply(list).to_dict()
        extra_rows = []
        for uid in short_users.index:
            current_items = set(result[result['user_id'] == uid]['item_id'])
            needed = TOPK - len(current_items)
            for item in als_dict.get(uid, []):
                if item not in current_items and needed > 0:
                    extra_rows.append({'user_id': uid, 'item_id': item})
                    current_items.add(item)
                    needed -= 1
        if extra_rows:
            result = pd.concat([result, pd.DataFrame(extra_rows)],
                               ignore_index=True)
    else:
        print("  All users have exactly 10 recommendations ✓")

    # ── 6. 검증 ───────────────────────────────────────────────────────────
    n_users   = result['user_id'].nunique()
    n_rows    = len(result)
    expected  = n_users * TOPK

    print(f"\n검증:")
    print(f"  총 유저 수 : {n_users:,}")
    print(f"  총 행 수   : {n_rows:,}  (expected {expected:,})")

    assert n_rows == expected, f"행 수 불일치: {n_rows:,} != {expected:,}"
    assert result.groupby('user_id').size().eq(TOPK).all(), \
        "일부 유저 추천 수 부족"
    print("  검증 통과 ✓")

    # ── 7. 저장 ───────────────────────────────────────────────────────────
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    fname    = f"output_ensemble_als{args.als_weight}_sas{args.sasrec_weight}.csv"
    out_path = os.path.join(args.output_dir, fname)
    result.to_csv(out_path, index=False)

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()