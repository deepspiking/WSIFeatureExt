# Unified WSI Feature Extraction (cTransPath + RetCCL)

이 디렉토리는 WSI(.svs)에서 타일 PNG를 만들고, cTransPath 또는 RetCCL 모델을 선택해 feature를 추출합니다.

## Pipeline

```text
data_ex/svs/*.svs
  -> wsi_read.py
  -> data_ex/tiles/{slide}/tiles/*.png
  -> run_inference.py --model {ctranspath|retccl}
  -> features/{model}/{slide}.pt, features/{model}/{slide}.csv
```

## LFS

```bash
git lfs install
git lfs pull
```

`data_ex/`, `model/` 아래 대용량 파일은 LFS로 관리됩니다.

## Data layout

예시 동작은 repo 로컬의 `./data_ex` 기준으로 설명합니다.

- `./data_ex/svs`: 예시용 입력 WSI
- `./data_ex/tiles`: 예시용 타일 출력
- `./data_ex/proteome`: 예시용 proteome TSV / derived vectors
- `./features`: 예시용 feature 출력

실제 대용량 데이터는 repo 최상위에서 아래처럼 symlink를 만들어 두고, CLI 인자로 명시해서 사용하는 것을 권장합니다.

```bash
ln -s /data/workspace/ai2bio/data ./data
```

즉, 기본값은 `data_ex`이고, 실제 데이터셋 실행은 `--input-dir ./data/...`, `--output-dir ./data/...`, `--tiles-root ./data/...`처럼 override하는 방식입니다.

## 1) WSI -> PNG (통합 전처리)

```bash
python wsi_read.py \
  --input-dir ./data_ex/svs \
  --output-dir ./data_ex/tiles \
  --target-mpp 1.0 \
  --patch-size 224 \
  --stride 224 \
  --white-threshold 240 \
  --max-white-ratio 0.75 \
  --black-threshold 15 \
  --max-black-ratio 0.10
```

동작 기준:
- cTransPath 방식처럼 level 0 고정 + MPP 기반 샘플링
- 저장 PNG는 항상 224x224로 통일
- RetCCL 기준의 엄격한 white/black 비율 필터 적용
- 경계 partial tile은 최소화(완전 윈도우 기준)

## 2) PNG -> Feature (모델 선택)

### cTransPath

```bash
python run_inference.py \
  --model ctranspath \
  --model-path ./model/ctranspath.pth \
  --svs-dir ./datas/svs \
  --tiles-root ./datas/tiles \
  --features-out ./features
```

### RetCCL

```bash
python run_inference.py \
  --model retccl \
  --model-path ./model/retccl.pth \
  --svs-dir ./datas/svs \
  --tiles-root ./datas/tiles \
  --features-out ./features
```

## Real dataset example

실제 데이터셋을 symlink된 `./data` 아래에서 돌릴 때는 기본값 대신 명시적으로 경로를 넘기면 됩니다.

```bash
python wsi_read.py \
  --input-dir ./data/CPTAC-BRCA_v1/BRCA \
  --output-dir ./data/CPTAC-BRCA_v1_feat/BRCA/tiles

python run_inference.py \
  --model ctranspath \
  --tiles-root ./data/CPTAC-BRCA_v1_feat/BRCA/tiles \
  --features-out ./data/CPTAC-BRCA_v1_feat/BRCA/ctranspath

python run_inference.py \
  --model retccl \
  --tiles-root ./data/CPTAC-BRCA_v1_feat/BRCA/tiles \
  --features-out ./data/CPTAC-BRCA_v1_feat/BRCA/retccl
```

Proteome TSV 추출 스크립트는 예시용 `data_ex`가 아니라 실제 TSV 경로를 `--input`으로 넘겨서 쓰는 유틸입니다.

## Proteome unshared vector extraction

기본 예시는 repo-local `data_ex/proteome` 아래의 TSV를 읽어, aliquot별 `10491` 길이 벡터 dict를 pickle로 저장합니다.

규칙:
- `Unshared Log Ratio` 컬럼만 사용
- `Mean`, `Median`, `StdDev` 행 제외
- 같은 aliquot의 `.1/.2/.3` 버전은 가장 큰 suffix만 선택
- `RetroIR.1`, `RetroIR.2` 제외
- 저장 key는 버전 suffix를 제거한 base aliquot id

```bash
python extract_unshared_proteome_vectors.py
```

산출물:
- `data_ex/proteome/CPTAC2_Breast_Prospective_Collection_BI_Proteome_unshared_vectors.pkl`
- `data_ex/proteome/CPTAC2_Breast_Prospective_Collection_BI_Proteome_unshared_vectors.selection.csv`
- `data_ex/proteome/CPTAC2_Breast_Prospective_Collection_BI_Proteome_unshared_vectors.genes.txt`

출력:
- `features/ctranspath/{slide}.pt`, `features/ctranspath/{slide}.csv`
- `features/retccl/{slide}.pt`, `features/retccl/{slide}.csv`

## Utilities

```bash
python inspect_features.py
python viewer.py
```
