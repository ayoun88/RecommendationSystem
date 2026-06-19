from recbole.quick_start.quick_start import load_data_and_model
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from recbole.utils.case_study import full_sort_topk
from tqdm import tqdm
import argparse
import json

from utils import *

TOPK = 10


def build_cart_candidates(train, test_start='2020-03-01', window_days=30):
    """
    test 시작일 기준 window_days 이내에
    cart 담았지만 purchase 안 한 아이템을 유저별로 추출.

    반환: {user_idx: [item_idx, ...]}  최신 cart 순 정렬
    """
    test_start   = pd.Timestamp(test_start, tz='UTC')
    window_start = test_start - pd.Timedelta(days=window_days)

    cart_df = train[
        (train['event_type'] == 'cart') &
        (train['event_time'] >= window_start) &
        (train['event_time'] <  test_start)
    ].copy()

    purchased_pairs = set(zip(
        train[train['event_type'] == 'purchase']['user_idx'],
        train[train['event_type'] == 'purchase']['item_idx'],
    ))
    cart_df['is_purchased'] = cart_df.apply(
        lambda r: (r['user_idx'], r['item_idx']) in purchased_pairs, axis=1
    )
    cart_not_bought = (cart_df[~cart_df['is_purchased']]
                       .sort_values('event_time', ascending=False))

    cart_candidates = (cart_not_bought
                       .groupby('user_idx')['item_idx']
                       .apply(list).to_dict())
    return cart_candidates


def build_category_popular(train, topk=50):
    """
    서브카테고리별 인기 아이템 사전 계산.
    purchase 기준 → 없으면 view 기준으로 fallback.

    반환: {category_code: [item_idx, ...]}  인기 순
    """
    purchase_df = train[train['event_type'] == 'purchase']
    cat_popular = (purchase_df
                   .groupby(['category_code', 'item_idx']).size()
                   .reset_index(name='cnt')
                   .sort_values(['category_code', 'cnt'], ascending=[True, False])
                   .groupby('category_code')['item_idx']
                   .apply(list).to_dict())
    return cat_popular


def build_user_last_category(train):
    """
    유저별 마지막으로 view한 서브카테고리.
    반환: {user_idx: category_code}
    """
    view_df = train[train['event_type'] == 'view'].sort_values('event_time')
    user_last_cat = (view_df.groupby('user_idx')['category_code']
                     .last().to_dict())
    return user_last_cat


def get_coldstart_items(uid, cart_candidates, user_last_cat,
                        cat_popular, global_popular, topk=10):
    """
    Cold start 유저 추천 우선순위:
      1순위) 2월 cart 담았지만 구매 안 한 아이템
      2순위) 마지막으로 본 서브카테고리 인기 아이템
      3순위) 전체 인기 아이템
    """
    result   = []
    item_set = set()

    # 1순위: cart 아이템
    for item in cart_candidates.get(uid, []):
        if item not in item_set:
            result.append(item)
            item_set.add(item)
        if len(result) == topk:
            return result

    # 2순위: 서브카테고리 인기 아이템
    last_cat = user_last_cat.get(uid)
    for item in cat_popular.get(last_cat, []):
        if item not in item_set:
            result.append(item)
            item_set.add(item)
        if len(result) == topk:
            return result

    # 3순위: 전체 인기 아이템
    for item in global_popular:
        if item not in item_set:
            result.append(item)
            item_set.add(item)
        if len(result) == topk:
            return result

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", default="train.parquet", type=str)
    parser.add_argument("--data_dir",      default="../data/",      type=str)
    parser.add_argument("--output_dir",    default="../output/",    type=str)
    parser.add_argument("--model_file",    default="./saved/SASRec.pth", type=str)
    parser.add_argument("--seed",          default=42, type=int)
    parser.add_argument(
        "--cart_window_days", type=int, default=30,
        help="cart fallback 기간 (일). 7=마지막 1주, 14=2주, 30=2월 전체"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. 데이터 로드 
    print("Loading data ...")
    train = pd.read_parquet(os.path.join(args.data_dir, args.train_dataset))
    train['event_time'] = pd.to_datetime(train['event_time'])
    train = train.sort_values(by=['user_session', 'event_time'])

    # 2. 인덱스 매핑 (JSON 기반 — 버그 수정)
    with open(os.path.join(args.data_dir, 'user2idx.json'), 'r') as f:
        user2idx = json.load(f)
    with open(os.path.join(args.data_dir, 'item2idx.json'), 'r') as f:
        item2idx = json.load(f)

    # idx2user, idx2item을 JSON 역변환으로 생성 (enumerate 재생성 버그 방지)
    idx2user = {int(v): k for k, v in user2idx.items()}
    idx2item = {int(v): k for k, v in item2idx.items()}

    train['user_idx'] = train['user_id'].map(user2idx)
    train['item_idx'] = train['item_id'].map(item2idx)

    users = defaultdict(list)
    for u, i in zip(train['user_idx'], train['item_idx']):
        users[u].append(i)

    # 3. Cold start 보조 데이터 사전 구축
    print("Building cold start helpers ...")

    # 1순위: 2월 cart 미구매 아이템
    cart_candidates = build_cart_candidates(
        train, window_days=args.cart_window_days
    )
    print(f"  Cart candidate users  : {len(cart_candidates):,}명")

    # 2순위: 서브카테고리별 인기 아이템
    cat_popular = build_category_popular(train)
    print(f"  Category popular dict : {len(cat_popular):,}개 카테고리")

    # 유저별 마지막 view 카테고리
    user_last_cat = build_user_last_category(train)
    print(f"  User last category    : {len(user_last_cat):,}명")

    # 3순위: 전체 인기 아이템 (item_idx 기준)
    global_popular = (train[train['event_type'] == 'purchase']
                      .groupby('item_idx').size()
                      .sort_values(ascending=False)
                      .index.tolist())
    # purchase 없는 아이템도 커버
    all_pop = (train.groupby('item_idx').size()
               .sort_values(ascending=False).index.tolist())
    global_popular = global_popular + [x for x in all_pop
                                       if x not in set(global_popular)]
    print(f"  Global popular pool   : {len(global_popular):,}개 아이템")

    # 4. SASRec 모델 로드
    config, model, dataset, _, _, test_data = load_data_and_model(
        model_file=args.model_file,
    )
    print('Model load complete')

    # 5. 추론 
    result       = []
    n_model      = 0   # SASRec 모델로 추론한 유저
    n_coldstart  = 0   # cold start fallback 유저
    n_cart_used  = 0   # cart fallback이 실제로 쓰인 유저
    n_cat_used   = 0   # 카테고리 fallback이 쓰인 유저

    for uid in tqdm(users):
        # SASRec에 포함된 유저 → 모델 추론
        if str(uid) in dataset.field2token_id['user_idx']:
            recbole_id = dataset.token2id(dataset.uid_field, str(uid))
            topk_score, topk_iid_list = full_sort_topk(
                [recbole_id], model, test_data,
                k=TOPK, device=config['device']
            )
            predicted_item_list = dataset.id2token(
                dataset.iid_field, topk_iid_list.cpu()
            )
            predicted_item_list = list(map(int, predicted_item_list[-1]))
            n_model += 1

        # Cold start 유저 → 3단계 fallback
        else:
            has_cart = uid in cart_candidates
            has_cat  = user_last_cat.get(uid) in cat_popular

            predicted_item_list = get_coldstart_items(
                uid, cart_candidates, user_last_cat,
                cat_popular, global_popular, topk=TOPK
            )
            n_coldstart += 1
            if has_cart:
                n_cart_used += 1
            elif has_cat:
                n_cat_used += 1

        for iid in predicted_item_list:
            result.append((idx2user[uid], idx2item[iid]))

    # 6. 검증 & 저장
    print(f"\n추론 완료:")
    print(f"  SASRec 모델 유저    : {n_model:,}명")
    print(f"  Cold start 유저     : {n_coldstart:,}명")
    print(f"    └ cart 활용       : {n_cart_used:,}명")
    print(f"    └ 카테고리 활용   : {n_cat_used - n_cart_used:,}명")
    print(f"    └ 전체 인기 활용  : {n_coldstart - n_cart_used - (n_cat_used - n_cart_used):,}명")

    result_df = pd.DataFrame(result, columns=["user_id", "item_id"])

    expected = len(users) * TOPK
    assert len(result_df) == expected, \
        f"행 수 불일치: {len(result_df):,} != {expected:,}"
    assert result_df['user_id'].nunique() == len(users), \
        "유저 수 불일치"
    assert result_df.groupby('user_id').size().eq(TOPK).all(), \
        "일부 유저 추천 수 부족"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    out_path = os.path.join(args.output_dir, "output_sasrec_3step_fallback.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(f"  총 행 수: {len(result_df):,}  ✓")


if __name__ == "__main__":
    main()