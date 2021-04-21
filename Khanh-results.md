| Task | Model | Method/Configuration | Train | Validation | Test |  
| --- | --- | --- | --- | --- | --- |  
| Audio | [BaseTimmModel](libs/models/baseline.py) | [EfficientNet B2 (NoisyStudent pretrained)](https://pastebin.com/TQGUPUdi) | 39.88 | 37.67 | **39.54** |  
| Audio | [BaseTimmModel](libs/models/baseline.py) | [SEResnext50](https://pastebin.com/DmFbzYXm) | 40.60 | 38.34 | **41.63** |  
| Audio | [BaseTimmModel](libs/models/baseline.py) | [EfficientNet B2 (Adversarial pretrained)](https://pastebin.com/w18ZFitq) | 49.54 | 38.77 | **41.97** |  
| Multimodal Fusion | [FusionModel](libs/models/fusion/multimodal_fusion.py) | Concatenation | 73.222 | 64.716 | **66.152** |  
| Multimodal Fusion | [FusionModel](libs/models/fusion/multimodal_fusion.py) | Multi-modal Factorized Bilinear Pooling | 74.09 | 64.85 | **66.054** |  
| Context-aware Recognition | [ContextAwareModel](libs/models/context_aware.py) | 2 utterances (MBP + 1 Transformer block) | 73.45 | 64.208 | **65.432** |  
| Context-aware Recognition | [ContextAwareModel](libs/models/context_aware.py) | 3 utterances (MBP + 1 Transformer block) | 73.658 | 64.552 | **65.806** |  
| Context-aware Recognition | [ContextAwareModel](libs/models/context_aware.py) | 5 utterances (MBP + 6 Transformer block) | 75.024 | 65.4 | **66.03** |  
| Context-aware Recognition | [ContextAwareModel](libs/models/context_aware.py) | 10 utterances (MBP + 6 Transformer block) | 73.44 | 62.98 | **65.16** |  