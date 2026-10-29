# CS228: AI-Based Authentication Against Presentation Attacks

This project studies facial authentication under presentation attacks (specifically spoofed images and replay attacks) by combining identity verification with liveness-style presentation attack detection (PAD). The core design is a two-stage pipeline: 1) compare a probe face against an enrolled reference identity, and 2) reject presentation attacks that attempt to spoof that identity.

The experimental plan for Checkpoint 1 follows the FaceNet-based recognition setup described below and retains both identity matching and PAD as required components.

The implementation details and end-to-end experiment are captured in the project notebook, while this README summarizes the design, methodology, and evaluation plan.

---

## Model choice

We will use FaceNet via the facenet-pytorch implementation, specifically InceptionResnetV1 pretrained on VGGFace2, together with MiniFASNetV2 for PAD.

Why this setup:
- FaceNet provides a documented PyTorch recognition pipeline with pretrained weights.
- InceptionResnetV1 produces 512-dimensional embeddings for face comparison.
- The implementation includes MTCNN face detection and uses 160 x 160 crop preprocessing.
- MiniFASNetV2 is used as the PAD model for determining whether a presented face is genuine.
- Both components are compatible in the same deep learning stack and are MIT-licensed.
- FaceNet reference: https://github.com/timesler/facenet-pytorch?utm_source=chatgpt.com
- CelebA-Spoof dataset reference: https://github.com/ZhangYuanhan-AI/CelebA-Spoof
- MiniFASNetV2 reference: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/README_EN.md

Then each FaceNet-aligned crop is fed to the PAD model and converted to a genuine score via a sigmoid/logit (logistic unit) output. FaceNet is used for identity verification via embedding distance; no new classifier needs to be trained for the CelebA-Spoof identities.

---

## Dataset and identity setup

We will use CelebA-Spoof for the benchmark, with both genuine and attack samples. The evaluation will preserve subject separation across train/validation/test splits, so that identities seen in one split do not leak into another.

For each evaluated identity, we reserve a genuine image as the enrollment reference. Then we construct the following verification trials:

- Genuine trial: another genuine image of the same person -> Accept
- Ordinary impostor trial: a genuine image of a different person -> Reject
- Presentation attack trial: a spoof depicting the enrolled person -> Reject

For attack trials, the spoof is compared against the person it depicts. This measures whether an attacker presenting the correct person’s appearance can pass authentication.

Each probe is represented by an embedding from FaceNet. We compute cosine similarity between the probe embedding and the enrollment reference embedding. The final decision is based on the combined rule:

$$
\text{Accept} = (\text{identity similarity} \geq \tau_{\mathrm{ID}}) \land (\text{genuine score} \geq \tau_{\mathrm{PAD}})
$$

where the identity threshold $\tau_{\mathrm{ID}}$ and PAD threshold $\tau_{\mathrm{PAD}}$ are chosen from validation data and frozen before testing.

FaceNet and MiniFASNet require their own preprocessing pipelines and crops. The FaceNet path uses the MTCNN-aligned 160 x 160 face crop, while MiniFASNet uses the PAD-ready crop and preprocessing expected by its implementation.

---

## Key research question

The project seeks to answer two intertwined questions:

1. How well does FaceNet alone perform ordinary identity verification, and how often do presentation attacks pass identity matching?
2. Does adding MiniFASNetV2 to the pipeline reduce successful attacks and increase rejection of genuine users without breaking usability?

---

## Checkpoint 1: comparison of two systems

We will compare these two systems on the same evaluation set:

### 1) FaceNet alone

This system makes verification decisions using only the identity similarity score. It establishes:
- ordinary verification performance,
- false matches for impostor attempts,
- how frequently attack trials pass identity matching.

### 2) FaceNet + MiniFASNetV2

This system requires both the identity check and the PAD check to pass. It measures:
- whether PAD reduces successful attacks,
- how much PAD increases rejection of genuine users,
- the combined attack success rate under the full authentication rule.

The combined system accepts an attempt only when both checks are successful. An attack succeeds against the combined system only if it passes both the identity threshold and the PAD threshold.

---

## Evaluation metrics

For the identity component, we report:
- FAR (False Acceptance Rate)
- FRR (False Rejection Rate)
- counts for every rate bucket, not only percentages

For the PAD component, we report:
- APCER (Attack Presentation Classification Error Rate)
- GPCER (Genuine Presentation Classification Error Rate)
- counts for each error rate

For the combined system, we report the combined attack success rate, defined as the fraction of attack trials that pass both identity verification and PAD. We also track:
- number of attack trials evaluated,
- number of attacks that pass identity only,
- number of attacks that pass PAD only,
- number of attacks that pass both checks,
- number of genuine trials rejected by the combined system.

The most important combined metric is the attack success rate under the full decision rule:

$$
\text{Combined Attack Success Rate} = \frac{\text{Number of attacks accepted by identity and PAD}}{\text{Number of all attacks}}
$$

---

## Experimental procedure

1. Split the dataset by subject to enforce separation between identities used for development and testing.
2. For each identity, select a genuine reference image for enrollment.
3. Build genuine, impostor, and spoof verification trials.
4. Compute FaceNet embeddings for all probes and reference images.
5. Compute cosine similarity for identity verification.
6. Compute PAD scores for the same probes using MiniFASNetV2.
7. Choose identity and PAD thresholds using validation data only.
8. Freeze the thresholds before final evaluation.
9. Report identity-only and combined-system metrics on the held-out test split.

---

## Pretraining overlap note

We will preserve strict subject separation in our local evaluation. However, the pretrained FaceNet model is trained on VGGFace2 and may include identities overlapping with the benchmark dataset. This overlap is not assumed to be absent and should be described as unverified unless we audit it directly. We will explicitly note this limitation in the final analysis.

---

## Summary of the proposed checkpoint

The selected experimental setup is:

| Component | Selection | Purpose |
| --- | --- | --- |
| Identity model | FaceNet / InceptionResnetV1, VGGFace2 weights | Compare the presented face with the enrolled identity |
| PAD model | MiniFASNetV2 | Determine whether the presentation appears genuine |
| Authentication rule | Identity match and PAD pass | Accept only when both checks pass |

This preserves the necessary recognition and antispoofing components while matching the project’s objective of studying AI-based authentication against presentation attacks.
