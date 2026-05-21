# cTransPath WSI Inference

CTransPath 모델을 사용해 Whole Slide Image(WSI)에서 patch feature를 추출하는 파이프라인입니다.

## 개요

```
datas/svs/*.svs
    │
    │  wsi_read.py
    │  (1.0 MPP 기준으로 patch 추출, background 필터링)
    ▼
datas/tiles/{slide_name}/tiles/*.png
    │
    │  run_inference.py
    │  (CTransPath → 768차원 feature 추출)
    ▼
features/{slide_name}.pt   — PyTorch tensor (N_tiles × 768)
features/{slide_name}.csv  — 동일 데이터, CSV 형식
```

## 빠른 시작

```bash
# 0. Git LFS 설치 (최초 1회)
git lfs install

# 1. 가상환경 생성 및 활성화 (Python 3.11 필요)
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. WSI에서 patch 이미지 생성
python wsi_read.py

# 4. 모델 inference → feature 추출
python run_inference.py

# 5. 결과 확인 (GUI 뷰어)
python viewer.py
```

## 디렉토리 구조

```
cTransPathInference/
├── datas/
│   ├── svs/          # 입력 WSI 파일 (.svs)
│   └── tiles/        # 추출된 patch 이미지 (자동 생성)
├── features/         # 추출된 feature 저장 위치 (자동 생성)
├── model/
│   └── ctranspath.pth  # 사전학습 모델 가중치
├── TransPath/        # CTransPath 모델 코드 (github.com/Xiyue-Wang/TransPath)
├── wsi_read.py       # WSI → patch 추출
├── run_inference.py  # patch → feature 추출
├── inspect_features.py  # 저장된 feature 정보 출력
├── viewer.py         # Tile-Feature GUI 뷰어
└── requirements.txt
```

## 설치

### 1. 모델 가중치 다운로드

[CTransPath Google Drive](https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view)에서 `ctranspath.pth`를 받아 `model/` 폴더에 넣습니다.

### 2. Git LFS 설치 (최초 1회)

```bash
git lfs install
```

> `datas/`(`.svs`, `.png`)와 `model/`(`.pth`)의 대용량 파일은 Git LFS로 관리됩니다.  
> LFS 없이 clone하면 포인터 파일만 받게 됩니다.

### 3. 가상환경 생성 및 패키지 설치

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **timm 버전 주의**: CTransPath는 `timm==0.5.4`에서만 정상 동작합니다.

### 4. WSI 파일 배치

`.svs` 파일을 `datas/svs/` 폴더에 넣습니다.

## 실행

### Step 1 — WSI에서 patch 추출

```bash
python wsi_read.py
```

- 슬라이드의 MPP를 자동으로 읽어 **1.0 MPP** 기준으로 patch 크기를 계산합니다.
  - 예: 슬라이드가 0.4942 MPP(20×)이면 `round(224 × 1.0 / 0.4942) = 453px` 크기로 추출
- 흰색 배경 patch는 자동으로 제외됩니다.
- 결과: `datas/tiles/{slide_name}/tiles/*.png`

설정값 (`wsi_read.py` 상단):

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `TARGET_MPP` | `1.0` | 목표 해상도 (CTransPath 권장값) |
| `PATCH_SIZE` | `224` | 모델 입력 크기 |
| `WHITE_THRESHOLD` | `240` | 배경 판단 밝기 기준 |
| `MIN_TISSUE_RATIO` | `0.05` | 조직 비율이 이 값 미만이면 배경으로 제외 |

### Step 2 — Feature 추출

```bash
python run_inference.py
```

- CTransPath 모델로 patch당 **768차원 벡터** 1개를 추출합니다.
- GPU(CUDA), Apple Silicon(MPS), CPU 순으로 자동 선택됩니다.
- 결과:
  - `features/{slide_name}.pt` — PyTorch 형식 (학습/분석용)
  - `features/{slide_name}.csv` — CSV 형식 (열람용)

CSV 컬럼: `tile_path`, `row`, `col`, `feat_0` ~ `feat_767`

### Feature 확인

```bash
# 터미널 출력
python inspect_features.py

# GUI 뷰어 (tile 클릭 → feature 시각화)
python viewer.py
```

## Feature 활용 예시

| 목적 | 방법 |
|------|------|
| Slide-level 분류 | ABMIL / CLAM으로 N_tiles × 768 → 1개 레이블 |
| Patch 분류 | 768차원 벡터에 Linear classifier 적용 |
| 조직 구조 시각화 | UMAP + k-means 클러스터링 후 슬라이드 위에 색 표시 |
| 단백질 발현 예측 | TCGA RPPA 데이터와 연계한 regression |
| 유사 patch 검색 | 벡터 간 cosine similarity |

## 참고

- 논문: [CTransPath (Medical Image Analysis, 2022)](https://www.sciencedirect.com/article/pii/S1361841522002043)
- 원본 코드: [github.com/Xiyue-Wang/TransPath](https://github.com/Xiyue-Wang/TransPath)
