# Localized Egocentric Vision

This repository studies efficient, causal action recognition in egocentric videos.
The current experiment uses the EGTEA dataset and asks:

> Does temporal context improve action recognition compared with classifying each sampled frame independently?

The model used is a causal transformer: its prediction at `t` may use frames up to `t - 1`, but never future frames.

## Current pipeline

```text
video frames
    ↓ sample every 15 frames
DINOv2-small (frozen pretrained encoder)
    ↓ one 384-dimensional vector per sampled frame
frame classifier or causal Transformer
    ↓
action prediction
```

DINOv2 is not trained in this experiment. Pre-trained weights are taken and are used to create vision embeddings for our video frames. It converts each RGB frame into a visual feature vector. The trainable models operate on those saved vectors, making repeated experiments faster and separating visual encoding from temporal modeling.

The implementation also supports a restricted-input path designed to study efficient, relevance-based visual encoders. A hand-region annotation is converted into a clipped bounding box, expanded by a controlled margin when required, and applied before DINOv2 inference. The encoder therefore spends its fixed visual-compute budget on pixels most directly associated with the hand-object interaction instead of repeatedly processing (relatively) irrelevant background, while preserving the same encoder weights, preprocessing interface, temporal model, and classifier. This creates a controlled representation-level comparison: any difference can be attributed to the visual field presented to the encoder rather than to a larger model or a different downstream architecture. The current restricted path is deliberately an oracle-input analysis that measures the potential value of relevant context without claiming automatic hand detection.

Evaluation uses leave-one-subject-out testing: each result trains on the other subjects and tests on the held-out subject. The annotation scan produced 73 observed action names.

## Models compared

- **Frame baseline:** a linear classifier predicts the action from one DINOv2 feature at a time.
- **Causal Transformer:** a temporal model receives sequences of cached DINOv2 features with a context limit of 128 sampled timesteps and causal attention.

Both models use the same features, labels, subject split, and evaluation procedure. This makes the comparison about temporal context rather than a different visual encoder.

## Engineering details

The code separates video decoding, temporal alignment, visual encoding, feature storage, and temporal learning. This prevents a model experiment from silently changing the sampling or labeling policy.

- **Deterministic sampling:** frame indices are generated explicitly from the video frame count and stride. A stride of 15 samples every fifteenth frame and preserves the original indices in the cache.
- **Tensor contracts:** cached features have shape `[time, 384]`; labels and frame indices have shape `[time]`. Temporal training adds a batch dimension, producing `[batch, time, 384]` inputs and `[batch, time]` labels.
- **Provenance-aware caches:** every cache stores features, labels, original frame indices, and metadata such as video ID, FPS, stride, encoder ID, feature dimension, and input type.
- **Atomic cache commits:** feature data is first written to a temporary file in the destination directory and then moved into place with `Path.replace`. A partially written extraction therefore does not appear as a valid final cache. Existing caches can be reused, avoiding repeated DINOv2 inference.
- **Alignment checks:** cache loading validates required fields and matching time dimensions. Paired full/restricted caches also verify that frame indices and labels agree before comparison.
- **Subject-level isolation:** train/test separation occurs before flattening or windowing features, preventing frames from the held-out subject from entering training.
- **Causal attention:** the Transformer uses a triangular attention mask, so the logit at timestep `t` cannot use features from timesteps greater than `t`.

These mechanisms address different failure modes: corrupted artifacts, accidental temporal misalignment, inconsistent visual inputs, subject leakage, and future-information leakage.

## Current results

Results below are from the four held-out-subject runs, trained for 20 epochs.

| Held-out subject | Frame accuracy | Transformer accuracy | Frame balanced accuracy | Transformer balanced accuracy |
|---|---:|---:|---:|---:|
| S1 | 0.5011 | 0.5375 | 0.3547 | 0.4402 |
| S2 | 0.4819 | 0.5422 | 0.3482 | 0.4714 |
| S3 | 0.4275 | 0.5518 | 0.2918 | 0.4374 |
| S4 | 0.5499 | 0.6752 | 0.3708 | 0.5661 |
| **Macro mean** | **0.4901** | **0.5767** | **0.3414** | **0.4788** |

The Transformer is higher on this pilot for both reported metrics. 

### Restricted-input pilot

Using the same EGTEA clips, labels, subject splits, and causal Transformer, the oracle hand-restricted representation produced the following paired results:

| Held-out subject | Full-frame accuracy | Restricted accuracy | Full balanced accuracy | Restricted balanced accuracy |
|---|---:|---:|---:|---:|
| S1 | 0.3793 | 0.4483 | 0.3200 | 0.4200 |
| S2 | 0.2500 | 0.4038 | 0.2351 | 0.3550 |
| S3 | 0.4444 | 0.4222 | 0.3280 | 0.3753 |
| S4 | 0.4524 | 0.3810 | 0.3362 | 0.2931 |
| **Macro mean** | **0.3815** | **0.4138** | **0.3048** | **0.3608** |

In this experiment, focusing the encoder on the annotated hand region matched or exceeded the full-frame representation on macro mean accuracy and balanced accuracy. This is encouraging evidence that much of the visual field may be unnecessary for these hand-object actions, and motivates relevance-aware efficient encoders.

## Reproducing the comparison

After installing the project dependencies and preparing the cached GTEA features:

```bash
bash scripts/subject-test.sh
bash scripts/train-transformer.sh
python3 -m scripts.compare_baselines
```

The per-subject JSON results are written to:

```text
outputs/frame_baseline/
outputs/causal_transformer/
```

Generated data, feature caches, model weights, and experiment outputs should remain outside Git unless explicitly needed as small documentation artifacts.
