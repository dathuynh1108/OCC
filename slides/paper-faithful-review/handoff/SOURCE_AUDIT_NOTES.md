# Source audit notes — 14 September 2026

Scope: author papers/code and the public OCC snapshot were read. No new training,
feature-parity experiment or original-paper numeric reproduction was run here.

## PatchCore versus the local PatchScore

The referenced source method is **PatchCore**, not a standalone paper named
PatchScore. The author's [backbone loader](https://github.com/amazon-science/patchcore-inspection/blob/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/src/patchcore/backbones.py)
uses pretrained Wide ResNet-50-2 for `wideresnet50`. Its
[Quick Guide](https://github.com/amazon-science/patchcore-inspection/blob/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/README.md)
specifies layer2/layer3 and 1024/1024 embedding dimensions.

The extraction path in [patchcore.py](https://github.com/amazon-science/patchcore-inspection/blob/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/src/patchcore/patchcore.py)
and [common.py](https://github.com/amazon-science/patchcore-inspection/blob/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/src/patchcore/common.py)
uses 3x3 patch extraction, grid alignment and adaptive pooling of patch descriptors.
The [OCC extractor](https://github.com/dathuynh1108/OCC/blob/91223d63adc3289786bbdeff16d5feeb59aeb3d7/nbdbench/data.py)
instead averages spatial maps and concatenates to 1536-D. Same CNN weights do not
make the two feature pipelines identical.

The [paper](https://cdn.amazon.science/ec/c4/8f8fad644e45a044262ca9fb95c1/towards-total-recall-in-industrial-anomaly-detection.pdf)
Eq. (7) describes neighbor reweighting. The inspected normal author-code prediction
path ends in maximum patch score without that term. Exact-paper and released-code
reproduction are therefore separate targets, not names to interchange.

## Deep SVDD

The [original implementation](https://github.com/lukasruff/Deep-SVDD/tree/e20f18c8d0ad9dc01cad09fdf311bd861351a9ad)
uses Theano/Lasagne and is identified by the author as the code used for the paper.
The [author PyTorch implementation](https://github.com/lukasruff/Deep-SVDD-PyTorch/tree/1901612d595e23675fb75c4ebb563dd0ffebc21e)
is a later reference; its example epoch schedules differ from the paper description.
The [paper](https://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf), Section 4,
uses native image CNNs initialized from a normal-image DCAE, then trains the encoder.
A head on a frozen ImageNet CNN is a distinct adaptation. Native preprocessing,
BatchNorm, activation, no-bias and center handling all need source parity.

## SVDD

The [author toolbox](https://github.com/DMJTax/dd_tools/tree/efaaf04efae1f8be78906836a5d31547b48be7af)
provides Gaussian-kernel SVDD via a quadratic program. It is not a CNN pipeline.
This is a later MATLAB reference, not proof of a recovered 2004 experiment.
In Ruff et al.'s comparison, the shallow SVDD baseline includes PCA and supervised
holdout tuning; it must not be described as the old CNN-coreset median-bandwidth run.

## DROCC

The [CIFAR runner](https://github.com/microsoft/EdgeML/blob/81025fce8ba28707eabe72e11bf3987a8d745608/examples/pytorch/DROCC/main_cifar.py)
uses trainable LeNet and input-space adversarial search. The
[supplement](https://proceedings.mlr.press/v119/goyal20c/goyal20c-supp.pdf), Table 11,
gives different settings per CIFAR class. The image experiments use gamma=1.

The source runner passes `only_ce_epochs=0`; its parser's ascent-step count is not
forwarded to the trainer. The [trainer](https://github.com/microsoft/EdgeML/blob/81025fce8ba28707eabe72e11bf3987a8d745608/pytorch/edgeml_pytorch/trainer/drocc_trainer.py)
projects every ten ascent steps. It selects the best model on the provided
validation loader, while that runner passes its test loader. The handoff requires
explicit historical/test-selected and fixed-final-epoch labels, not hidden leakage
and not a silent evaluation change presented as exact reproduction.

## Limits and outcome

These observations establish source differences, not the cause of the poor v2
scores. The new rerun may be better or worse. Existing measured v2 tables and
all NBD ablations remain unchanged, and no native result was filled in.
