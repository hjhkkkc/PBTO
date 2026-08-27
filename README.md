# PBTO reproduction scaffold

This repository is a **best-effort, from-paper implementation** of *Persistent Backdoor Attacks in Class-Incremental Learning via Structural Invariant Anchoring*. It is restricted to local, controlled benchmark experiments and contains no networking, deployment, or real-system attack automation.

The implementation covers:

- class-incremental iCaRL with an expandable CIFAR ResNet-18;
- a fixed exemplar budget and feature-mean herding;
- clean and poisoned proxy parameter trajectories;
- universal additive trigger optimization over all trajectory checkpoints;
- Gram-matrix structural anchoring at configurable residual stages;
- iterative poisoned-trajectory/trigger refinement;
- one-time Task-1 poisoning, task-wise BA and ASR;
- component, trajectory-length, proxy-size, replay-memory and lambda sweeps;
- CKA, Grassmann distance and initial-subspace variance retention;
- channel-level Fisher/parameter-drift overlap analysis.

It does **not** claim bit-for-bit equality with the unavailable author code. Several material settings are not reported in the paper, and the exact public-image queries, proxy classes, filters and DDPM checkpoint are absent.

## 1. Paper-specified settings implemented directly

The paper specifies the following main protocol, which is reflected in `configs/cifar100_pbto_paper_candidate.yaml`:

| Setting | Value |
|---|---:|
| CIFAR-100 victim tasks | 10 × 10 classes |
| Victim learner | iCaRL + ResNet-18 |
| Exemplar memory | 2,000 |
| Poisoning stage | Task 1 only |
| Poison rate | 5% of Task-1 training set |
| Trigger constraint | `L_inf <= 8/255` |
| Proxy trajectory | 5 tasks × 5 classes |
| Proxy images | 5,000 per class |
| Trigger optimizer | PGD |
| Structural weight | `lambda = 1` |
| Repetitions | 3 seeds/class orders/initializations |

## 2. Settings that remain assumptions

The paper does not disclose the following values. They are explicit YAML fields so they can be swept rather than silently assumed:

- CIL training epochs and learning-rate schedule;
- PGD step count, step size and trigger batch size;
- exact Gram-matrix normalization/reduction;
- whether the target reference is one image or a class statistic;
- exact anchor-layer indexing;
- refinement threshold `tau`, tolerance and `Tmax`;
- exact proxy class list, web queries, filtering rules and DDPM model;
- whether reported iCaRL evaluation uses NME or classifier logits.

The supplied candidate uses 70 CIL epochs, PGD 200 × `1/255`, mean normalized Gram matrices, NME inference, and a literal `layer3` interpretation of “penultimate residual block/stage.” `configs/cifar100_pbto_layer2.yaml` provides the shallower alternative suggested by the paper's subspace-stability table.

## 3. Installation

```bash
cd pbto_reproduction
python -m venv .venv
source .venv/bin/activate
pip install -e ".[analysis,test]"
pytest
```

Python 3.10+ and a CUDA GPU are recommended. The full refinement loop is computationally expensive.

## 4. Proxy data layout

For the paper-oriented configuration, prepare public or generated proxy images as an `ImageFolder`:

```text
proxy_data/cifar100/
├── apple/                 # semantic target class; name set in YAML
│   ├── 000001.jpg
│   └── ...
├── proxy_class_02/
├── proxy_class_03/
└── ...                    # at least 25 balanced classes
```

A `train/<class>/...` and `val/<class>/...` layout is also accepted. The target proxy class is moved into proxy Task 1 and receives surrogate label 0. The victim target class is likewise moved into victim Task 1 and receives incremental label 0.

`configs/smoke_cifar10.yaml` uses CIFAR-100 as a debug proxy so the code path can be checked without collecting a proxy folder. That setup is **not** a faithful threat-model reproduction.

## 5. Run order

### 5.1 Verify the clean CIL baseline

```bash
python scripts/run_clean_cil.py \
  --config configs/clean_cifar100.yaml
```

Outputs include one checkpoint per task and `clean_metrics.csv`.

### 5.2 Run the core PBTO experiment

```bash
python scripts/run_pbto.py \
  --config configs/cifar100_pbto_paper_candidate.yaml
```

A cheaper engineering check is:

```bash
python scripts/run_pbto.py \
  --config configs/smoke_cifar10.yaml
```

Useful command-line overrides:

```bash
# One refinement round and 50 PGD steps
python scripts/run_pbto.py --config configs/cifar100_pbto_paper_candidate.yaml \
  --set refinement.max_rounds=1 \
  --set trigger.steps=50

# Use the shallower Module-2 anchor
python scripts/run_pbto.py --config configs/cifar100_pbto_paper_candidate.yaml \
  --set trigger.anchor_layers='[layer2]'

# Evaluate with classifier logits rather than iCaRL NME
python scripts/run_pbto.py --config configs/cifar100_pbto_paper_candidate.yaml \
  --set evaluation.inference=logits
```

### 5.3 Analyze structural stability

```bash
python scripts/analyze_subspace.py \
  --run-dir outputs/clean_cifar100_seed0 \
  --layers layer1 layer2 layer3 layer4 \
  --max-samples 1000
```

This produces `subspace_metrics.csv` with Task-1-referenced linear CKA, normalized geodesic Grassmann distance and projection variance retention. The exact Grassmann normalization is stated in `src/pbto_repro/subspace.py` because the paper does not define it.

### 5.4 Analyze critical-neuron stability

```bash
python scripts/analyze_neuron_stability.py \
  --run-dir outputs/clean_cifar100_seed0 \
  --fractions 0.01 0.05 0.10
```

The code uses convolution-output channels as “neurons,” diagonal Fisher aggregated per output channel, and relative channel-weight drift from the Task-1 checkpoint. The paper does not disclose these implementation choices, so this output should be treated as a trend reproduction rather than an exact Figure-1 reconstruction.

### 5.5 Launch ablations

```bash
# Table-7-style component ablation
python scripts/run_ablation.py \
  --config configs/cifar100_pbto_paper_candidate.yaml \
  --mode components \
  --output-root outputs/ablation_components

# Other modes: trajectory, lambda, memory, proxy_size
```

Use `--dry-run` to print commands without executing them.

## 6. Output structure

A PBTO run writes:

```text
outputs/<run>/
├── resolved_config.yaml
├── class_orders.json
├── proxy_clean_trajectory/task_*.pt
├── pbto_trigger/
│   ├── trigger_round_*.pt
│   ├── trigger_round_*.csv
│   ├── refinement.csv
│   └── trigger_final.pt
├── victim_trajectory/task_*.pt
└── victim_taskwise_metrics.csv
```

`victim_taskwise_metrics.csv` reports:

- `benign_accuracy`: accuracy over all classes seen so far;
- `asr`: fraction of non-target test images classified as the target after adding the trigger;
- `target_clean_accuracy`: clean accuracy on the target class, useful for detecting trivial target bias.

The trigger is added to raw `[0, 1]` pixels, clipped, and only then normalized inside the network.

## 7. Correspondence between equations and code

| Paper component | Implementation |
|---|---|
| Proxy trajectory `Omega={theta_1,...,theta_M}` | `trajectory.train_icarl_trajectory` |
| Trajectory CE `L_t` | `trigger.optimize_universal_trigger` |
| Gram anchoring `L_s` | `trigger.gram_matrix` and reference Gram statistics |
| PGD under `L_inf` | `UniversalAdditiveTrigger.project_` and signed descent |
| Alternating refinement | `pbto.run_iterative_refinement` |
| Dataset replacement poisoning | `data.PoisonedDataset` |
| iCaRL exemplar replay | `memory.ExemplarMemory` and `trainer.ICaRLTrainer` |
| Task-wise BA/ASR | `metrics.py` and `scripts/run_pbto.py` |

## 8. Important interpretation notes

1. **Proxy semantics matter.** A random proxy target class is not equivalent to a public-data proxy for the victim target semantic class.
2. **Literal layer indexing is ambiguous.** In torchvision ResNet-18, `layer3` is the penultimate residual stage, but the reported stability numbers favor `layer1`/`layer2`. Run both.
3. **Refinement cost can dominate.** The paper's plot shows saturation around 50 rounds, but retraining a full five-task iCaRL trajectory 50 times is expensive. Begin with `max_rounds=1–3` to validate the implementation.
4. **Exact table values are not guaranteed.** Treat the main acceptance criteria as qualitative: shallow layers are more stable; static triggers decay; trajectory-only and alignment-only improve persistence; their combination performs best.
5. **Advanced baselines and defenses are not reimplemented.** WaNet, DRUPE, LTB, ISSBA, LIRA, WaveAttack, NAD, BTI-DBF and REFINE should be integrated from their official repositories under the same task split and evaluation code. Re-creating all of them from prose would introduce additional uncontrolled deviations.

## 9. Controlled-use scope

Use this repository only for authorized robustness evaluation on local datasets and models. The code intentionally omits integrations for external services, deployed models, data-collection automation and stealthy operational delivery.
