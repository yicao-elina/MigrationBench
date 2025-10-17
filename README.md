# MigrationBench 🧠⚙️
**Benchmarking Specialist vs. Generalist ML Force Fields via Migration Pathways**

> Official implementation of the NeurIPS AI4Mat 2025 paper  
> *"Migration as a Probe: A Generalizable Benchmark Framework for Specialist vs. Generalist Machine-Learned Force Fields"*

---

## 🧩 Overview
**MigrationBench** provides a generalizable benchmarking pipeline for evaluating **machine-learned force fields (MLFFs)** based on their ability to reproduce **migration properties** such as energy barriers, transition states, and pathway stability.

The framework uses **migration as a physical probe** to distinguish between *specialist* (fine-tuned) and *generalist* (foundation) models, uncovering how training paradigms shape chemical intuition and kinetic reliability.

---

## 🚀 Key Features
- **Unified Pipeline** for AIMD → NEB → MLFF → Error Analysis  
- **Model-Agnostic Interface** supporting MACE, NequIP, SchNet, M3GNet, and others  
- **Automatic NEB Diagnostics** detecting unstable (explosive) fine-tuned models  
- **Latent Space Visualization** revealing structure-property divergence  
- **Extensible Benchmarking API** for community submissions and model comparison

---

## 🧠 Core Concept
| Concept | Description |
|----------|--------------|
| **Migration as Probe** | Use atomic migration pathways to evaluate MLFF robustness |
| **Specialist vs. Generalist** | Compare fine-tuned vs. pretrained model paradigms |
| **Failure Analysis** | Identify “explosive” or non-convergent NEB behaviors |
| **Latent Diagnostics** | Track learned representations across model variants |

---

## 🧰 Installation
```bash
git clone https://github.com/yicao-elina/MigrationBench.git
cd MigrationBench
pip install -e .
````

Dependencies (Python ≥3.9):

* ASE
* pymatgen
* MACE / NequIP / SchNetPack
* numpy, pandas, matplotlib
* scipy, tqdm

---

## 📊 Quick Start

```python
from migrationbench import Benchmark

bench = Benchmark(
    material="Sb2Te3",
    model="mace-specialist",
    migration_path="data/neb/path_1"
)
bench.run()
bench.plot_energy_profile()
bench.export_metrics("results/barrier_comparison.csv")
```

---

## 🧪 Example Outputs

* NEB energy barrier plots comparing DFT vs. MLFFs
* Convergence diagnostics for migration stability
* Latent representation maps across models
* Model ranking by physical coherence

---

## 🔬 Citation

If you use this benchmark, please cite:

>Cao, Yi, and Paulette Clancy. "Migration as a Probe: A Generalizable Benchmark Framework for Specialist vs. Generalist Machine-Learned Force Fields." arXiv preprint arXiv:2509.00090 (2025).

---

## 🤝 Contributing

We welcome contributions!
You can:

* Submit your MLFF results using our standard output format
* Propose new migration systems or evaluation metrics
* Share feedback via [Issues](https://github.com/YiCao-JHU/MigrationBench/issues)

---

## 📧 Contact

Yi Cao — Johns Hopkins University
📮 [ycao73@jh.edu](mailto:ycao73@jh.edu)

---

### 🌍 “Migration reveals what metrics miss.”



---
