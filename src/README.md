# E-commerce item Recommendation Baseline Code

이커머스 상품 추천 대회를 위한 베이스라인 코드입니다.

## Installation

```
python 3.10 이상의 환경에서 진행해주시길 바랍니다.
pip install -r requirements.txt
```

## How to run

### MODEL

#### ALS

1. train ALS model and inference
   ```
   python train_als.py
   ```

#### SASRec

1. Prepare datset for using Recbole library
   ```
   python recbole_dataset.py
   ```

2. Train SASRec model 
   ```
   python train_sasrec.py
   ```

3. Infrence with trained sasrec model
   2번에서 생성된 model 파일의 상대 경로가 './saved/SASRec.pth'라면
   ```
   python inference_sasrec.py --model_file ./saved/SASRec.pth
   ```