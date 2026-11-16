# WISDM Wearable Human Activity Recognition

**Course:** Deep Learning (AI403) – Fall 2026, Pharos University in Alexandria
**Team name:** TODO
**Team members:** TODO (names + IDs)
**Idea chosen:** #9 – Wearable human activity recognition (inertial sensors -> activity)

## 1. Task and best result
Classify 18 activities from windows of smartphone/smartwatch accelerometer and gyroscope signals (200 timesteps x 12 channels).

Best result: TODO (model, macro-F1 mean ± SD over 3 seeds)

## 2. Models
| Role | Model | File |
|---|---|---|
| Baseline | Small MLP (2400-128-64-18) | `src/models/mlp.py` |
| Classic | 1D-CNN | `src/models/cnn1d.py` |
| Modern | Transformer | `src/models/transformer.py` |
| Advanced | ResNet1D (Residual Block written by us) | `src/models/resnet1d.py` |

## 3. Environment setup (Command Prompt)
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Dataset download and preprocessing
```cmd
python data\download.py
python data\preprocess.py
```

## 5. Training (one command per model)
```cmd
python src\train.py --config configs\mlp.yaml
python src\train.py --config configs\cnn1d.yaml
python src\train.py --config configs\transformer.yaml
python src\train.py --config configs\resnet1d.yaml
```

## 6. Evaluation
```cmd
python src\evaluate.py --config configs\resnet1d.yaml --checkpoint checkpoints\resnet1d_seed0.pt
```

## 7. Demo
```cmd
python demo\app.py
```

## 8. Hardware and expected runtimes
TODO

## 9. Credits
- WISDM dataset: TODO (source, version, licence)
- External code / pretrained models: TODO (cite and describe adaptation)
"# Deep_learning.wisdm-har" 
