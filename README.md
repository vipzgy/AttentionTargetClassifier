# AttentionTargetClassifier
implement "Attention Modeling for Targeted Sentiment"

## Requirements
 - python >= 3.5
 - pytorch == 0.3.1

## Configuration
`default.ini` is a configuration file, which contains all parameters of model, train, and test.

## Quick Start
0、prepare
```bash
git clone https://github.com/vipzgy/AttentionTargetClassifier
cd AttentionTargetClassifier
```
1、preprocess
```python
python preprocess.py
```
2、train
```python
python train.py
```