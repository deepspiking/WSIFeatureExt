# Unified WSI Feature Extraction (cTransPath + RetCCL)

이 디렉토리는 WSI(.svs)에서 타일 PNG를 만들고, cTransPath 또는 RetCCL 모델을 선택해 feature를 추출합니다.

## Pipeline

```text
datas/svs/*.svs
  -> wsi_read.py
  -> datas/tiles/{slide}/tiles/*.png
  -> run_inference.py --model {ctranspath|retccl}
  -> features/{slide}.{model}.pt, features/{slide}.{model}.csv
```

## LFS

```bash
git lfs install
git lfs pull
```

`datas/`, `model/` 아래 대용량 파일은 LFS로 관리됩니다.

## 1) WSI -> PNG (통합 전처리)

```bash
python wsi_read.py \
  --input-dir ./datas/svs \
  --output-dir ./datas/tiles \
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
python run_inference.py --model ctranspath --model-path ./model/ctranspath.pth
```

### RetCCL

```bash
python run_inference.py --model retccl --model-path ./model/retccl.pth
```

출력:
- `features/{slide}.ctranspath.pt`, `features/{slide}.ctranspath.csv`
- `features/{slide}.retccl.pt`, `features/{slide}.retccl.csv`

## Utilities

```bash
python inspect_features.py
python viewer.py
```
