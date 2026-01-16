## Voice Spoofing Generalization (ASVspoof → LibriSpeech)
This repository investigates whether a voice anti-spoofing model trained on the ASVspoof2019-LA benchmark can generalize to real-world speech outside the benchmark domain. A lightweight LFCC-CNN classifier is trained on labeled ASVspoof data and evaluated on both in-domain and out-of-domain speech. The model achieves near-perfect performance on ASVspoof but misclassifies natural speech from LibriSpeech, revealing a persistent gap between benchmark success and real-world deployment suitability.
## Dataset Overview
- Training / Dev: ASVspoof2019-LA (labeled)
- Evaluation: ASVspoof2019-LA Eval (unlabeled)
- Out-of-Domain Test: LibriSpeech Test-Clean (bonafide only)
## Background: Voice Spoofing & ASVspoof Benchmark
Modern speaker verification systems are vulnerable to voice spoofing attacks, including:
- neural text-to-speech (TTS)
- voice conversion (VC)
- replay attacks  
To support research, the ASVspoof Challenge provides standardized datasets that contain both bona fide (real human speech) and spoofed speech generated using known attack systems under controlled conditions.
We focus on ASVspoof2019-LA, which targets logical access attacks (i.e., TTS and VC systems). It contains:  
- training split (labeled)
- development split (labeled)
- evaluation split (unlabeled)  
ASVspoof is an excellent benchmark for studying spoof detection, but it does not represent natural unconstrained speech usage scenarios.
## Motivation
A model that performs well on ASVspoof may still fail in practice if it:
- rejects real human users,
- over-flags unfamiliar bona fide speech,
- or depends on dataset-specific artifacts.  
In real deployments (e.g., authentication, forensic verification, fraud prevention), false rejections of genuine users are just as damaging as failing to detect spoofing attacks.  
This project asks:  
### `Does a high-performing ASVspoof model recognize bona fide speech outside the ASVspoof domain?`
## Exploratory Data Analysis  
Basic EDA on ASVspoof2019-LA revealed:
- Imbalanced classes: Spoof ≫ bona fide
- Short utterances: Majority < 4 sec, enabling fixed-length inputs
- Spectral differences: Spoof audio shows smoother high-frequency structure; bona fide speech displays natural variation
- Single bona fide domain: Limited diversity compared to natural speech corpora
## Feature Extraction & Preprocessing
Speech audio was converted into fixed-dimensional LFCC feature maps suitable for CNN-based spoof detection. The pipeline included:
- Audio normalization: Waveforms normalized to consistent amplitude levels.
- Fixed-duration segmentation: All utterances were truncated or zero-padded to 4 seconds at 16 kHz,
  resulting in a consistent input length of 64,000 samples.
- LFCC feature extraction : Selected due to sensitivity to high-frequency synthesis artifacts.
Parameters:
  - Sampling Rate: 16 kHz
  - LFCC Coefficients: 60
  - FFT Size: 512
  - Window: 25 ms
  - Hop: 10 ms  
- Precomputation of LFCCs: All LFCC tensors were precomputed offline to avoid expensive on-the-fly transformations and accelerate training in Colab.
- PyTorch Dataset & DataLoaders  
Custom dataset classes returned (LFCC_tensor, label) pairs.
DataLoaders handled batching, shuffling, and collate functions for:
  - ASVspoof train/dev/eval splits
  - LibriSpeech test-clean out-of-domain evaluation
## Model Architecture
We implement a compact 5-layer CNN operating on LFCC tensors (1×60×T) with Conv-BN-ReLU blocks and a fully connected output layer trained with **BCEWithLogitsLoss + pos_weight**.
A learning rate scheduler **(ReduceLROnPlateau)** stabilizes late-stage training.
## Results  
Final evaluation was performed across three domains to measure both in-distribution benchmark performance and cross-corpus generalization:
| Evaluation Domain             | Ground Truth            | Result Summary                                                  | Interpretation                              |
|------------------------------|--------------------------|-----------------------------------------------------------------|----------------------------------------------|
| ASVspoof Dev (Labeled)       | Spoof + Bona fide        | **AUC:** 0.999965 • **EER:** 0.2344% • **ACC:** 99.83%          | Near-perfect in-domain separation            |
| ASVspoof Eval (Unlabeled)    | Spoof + Bona fide (hidden) | **71.96%** predicted spoof                                      | Retains spoof decision boundary; no collapse |
| LibriSpeech Test-Clean (OOD) | Bona fide only           | **59.6% FPR** (bonafide → spoof)                                | High false positives; poor OOD generalization |

## Key Observation:
The model performs exceptionally well on ASVspoof but misclassifies natural human speech from LibriSpeech, indicating that it learned dataset-specific spoofing cues rather than a general notion of human speech authenticity.

## Repo Structure
```
voice-spoofing-generalization/
├── README.md
├── report.pdf
├── presentation.pptx
├── notebooks/
│   ├── 1_Data_Wrangling_and_EDA.ipynb
│   ├── 2_Preprocessing_and_Modeling.ipynb
│   └── 3_OOD_Evaluation_LibriSpeech.ipynb
├── data/                 # (not included in repo)
  ├── asvspoof/
  └── librispeech/

```



