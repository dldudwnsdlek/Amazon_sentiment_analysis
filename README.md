# Amazon 영화 리뷰 감정 기반 추천 시스템

Amazon Movies & TV 카테고리의 대규모 리뷰 데이터를 분석하여, 사용자의 감정 상태를 반영한 딥러닝 기반 영화 추천 시스템을 구축한 캡스톤 디자인 프로젝트입니다.

---

## 프로젝트 개요

사용자가 남긴 리뷰 텍스트에서 감정을 추출하고, Neural Collaborative Filtering(NCF) 모델을 활용해 개인화된 영화를 추천합니다. 단순한 평점 기반 추천을 넘어, 감정·장르·연도 등 다양한 피처를 결합한 하이브리드 추천 시스템을 목표로 합니다.

---

## 데이터셋

| 항목 | 내용 |
|---|---|
| 원본 데이터 | Amazon Movies and TV 리뷰 (`movies.txt`) |
| 메타 데이터 | `meta_Movies_and_TV.json` (ASIN ↔ 영화 제목 매핑) |
| 원본 리뷰 수 | 약 7,911,696건 |
| 전처리 후 최종 리뷰 수 | 약 224,961건 |
| 고유 영화 수 | 11,012개 (제목 기준 10,511개) |

---

## 파이프라인

### 1. 데이터 파싱 및 청크 분할 (`movies_chunk_code.ipynb`)

- `movies.txt`를 한 줄씩 읽어 리뷰 단위로 파싱
- 100,000건씩 CSV 파일로 저장 (총 80개 청크: `movies_chunk_0.csv` ~ `movies_chunk_79.csv`)
- 컬럼: `product/productId`, `review/userId`, `review/profileName`, `review/helpfulness`, `review/score`, `review/time`, `review/summary`, `review/text`

### 2. 영화 제목 매핑 (`chunk_title_code.ipynb`)

- `meta_Movies_and_TV.json`에서 ASIN → 영화 제목 딕셔너리 생성
- 각 청크 CSV에 `movie_title` 컬럼 추가
- 결과물: `movies_chunk_*_with_title.csv`

### 3. 데이터 병합 및 정제 (`fianl_data_save.ipynb`)

- 80개 청크를 `pd.concat`으로 병합 (7,911,696건)
- 결측치 제거 후 1,262,842건으로 축소
- 불필요 컬럼 제거 및 컬럼명 변경
- 최종 컬럼: `item_id`, `rating`, `summary`, `text`, `title`
- 저장: `final_data.csv`

### 4. 텍스트 분석 및 감정 분류 (`text_analysis.ipynb`)

- `final_data.csv`에 장르·연도 정보 조인 → 229,655건
- **탐색적 데이터 분석(EDA)**
  - 평점 분포 시각화
  - 리뷰 수 상위 영화 및 평균 평점 분석
  - 영화 ID와 제목 수 불일치 원인 분석 (고전 영화 중복 등)
- **텍스트 전처리** (NLTK)
  1. 토큰화 (word_tokenize)
  2. 소문자 변환
  3. 불용어 제거 (영어 stopwords)
  4. 표제어 추출 (Lemmatization)
- **TF-IDF 벡터화**: 224,961건 × 182,038 어휘 행렬
- **감정 분류** (HuggingFace Transformers)
  - 모델: `bhadresh-savani/distilbert-base-uncased-emotion`
  - 배치 추론 (batch_size=32, truncation=512 토큰)
  - 6가지 감정 분류: 즐거움(joy), 슬픔(sadness), 분노(anger), 긴장(fear), 설렘(surprise), 편안함(love)
- 최종 저장: `text_data.csv` (224,961건)

### 5. 딥러닝 추천 모델 (`deep_code_class_after.ipynb`)

- **Neural Collaborative Filtering(NCF)** 기반 추천 시스템
- **입력 피처**
  - 사용자 ID 임베딩 (Embedding Layer)
  - 아이템 ID 임베딩 (Embedding Layer)
  - 장르 (One-Hot Encoding)
  - 출시 연도
  - 감정 피처 (refined_emotion, dominant_emotion)
  - 제목 임베딩
- **모델 아키텍처**: Embedding → Flatten → Concatenate → Dense + Dropout + BatchNormalization
- **버전별 개선**

  | 버전 | 특징 |
  |---|---|
  | 기본 모델 | NCF 기반 기초 구조 |
  | Safe 모델 | K-Fold 교차검증, 과적합 방지 |
  | Incremental 모델 | 자동 하이퍼파라미터 튜닝, RobustScaler |

- **평가 지표**: Precision@K, Recall@K, NDCG, F1-Score, AUC-ROC

---

## 기술 스택

| 분류 | 라이브러리 |
|---|---|
| 데이터 처리 | pandas, numpy |
| 머신러닝 | scikit-learn (TF-IDF, MinMaxScaler, RobustScaler, KFold) |
| 딥러닝 | TensorFlow / Keras |
| 자연어 처리 | NLTK (tokenize, stopwords, lemmatize), HuggingFace Transformers |
| 감정 분류 모델 | DistilBERT (`bhadresh-savani/distilbert-base-uncased-emotion`) |
| 시각화 | matplotlib, seaborn |

---

## 파일 구조

```
Amazon_sentiment_analysis/
├── movies_chunk_code.ipynb       # 원본 텍스트 파싱 및 청크 분할
├── chunk_title_code.ipynb        # ASIN → 영화 제목 매핑
├── fianl_data_save.ipynb         # 청크 병합 및 데이터 정제
├── text_analysis.ipynb           # EDA, 텍스트 전처리, 감정 분류
├── deep_code_class_after.ipynb   # NCF 딥러닝 추천 모델
└── 캡스톤디자인-최종보고서_pdf.pdf   # 최종 보고서
```

---

## 실행 방법

### 환경 설정

```bash
pip install pandas numpy scikit-learn tensorflow transformers nltk tf-keras
```

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 실행 순서

1. `movies_chunk_code.ipynb` — 원본 데이터 파싱
2. `chunk_title_code.ipynb` — 영화 제목 추가
3. `fianl_data_save.ipynb` — 데이터 병합 및 정제
4. `text_analysis.ipynb` — 텍스트 전처리 및 감정 분류
5. `deep_code_class_after.ipynb` — 추천 모델 학습 및 평가

> **참고**: 각 노트북의 파일 경로(`DATA_PATH`)를 로컬 환경에 맞게 수정해야 합니다.

---

## 주요 결과

- 원본 7,911,696건의 리뷰를 정제하여 224,961건의 고품질 데이터 구축
- DistilBERT 기반 감정 분류로 리뷰별 6가지 감정 레이블 자동 부여
- NCF 모델에 감정·장르·연도 피처를 결합한 하이브리드 추천 시스템 구현
- K-Fold 교차검증 및 하이퍼파라미터 튜닝을 통한 과적합 방지
