# Summer Research — Reading List

A curated list of papers and projects organized by research area.

---

## 🌐 World Models

### Foundational

| Paper | Authors | Year |
|-------|---------|------|
| **World Models** | David Ha, Jürgen Schmidhuber | 2018 |
| **Dream to Control (Dreamer)** | Hafner et al. | 2019 |
| **Mastering Atari with Discrete World Models (DreamerV2)** | Hafner et al. | 2020 |
| **Mastering Diverse Domains through World Models (DreamerV3)** | Hafner et al. | 2023 |
| **Dreamer-CDP** | — | 2026 |
| **R2-Dreamer** | — | 2026 |

**World Models (2018):** Classic VAE → RNN/MDN-RNN → Controller pipeline. Observations compressed into latent space; agent acts inside imagined rollouts.

**Dreamer (2019):** Actor-Critic trained purely in latent space via imagination rollouts with analytic gradients.

**DreamerV2 (2020):** Extends Dreamer with discrete latent spaces, categorical VAEs, and straight-through estimators.

**DreamerV3 (2023):** Focuses on scaling, generality, long-horizon planning, and sparse reward settings.

**Dreamer-CDP / R2-Dreamer (2026):** Recent extensions — details TBD.

---

### JEPA & Predictive Architectures

| Paper | Authors | Year |
|-------|---------|------|
| **JEPA (Joint-Embedding Predictive Architecture)** | Yann LeCun et al. | — |
| **LeWorldModel** | — | 2026 |

**JEPA:** LeCun's alternative to generative world models — avoids pixel-level prediction losses in favor of latent predictive embeddings.

**LeWorldModel (2026):** Stable end-to-end JEPA from pixels. Covers self-supervised predictive learning, latent geometry, and contrastive alternatives.

---

### Interactive World Generation

| Paper | Authors | Year |
|-------|---------|------|
| **Genie** | Google DeepMind | 2024 |
| **Genie 2 / Genie 3** | Google DeepMind | 2024+ |

**Genie (2024):** Generates interactive environments from videos, prompts, and images.

**Genie 2/3:** Advances persistent 3D worlds, longer memory, interactive exploration, and environment editing.

---

### Model-Based RL

| Paper | Authors | Year |
|-------|---------|------|
| **MuZero** | DeepMind | 2020 |

**MuZero:** Learns latent dynamics, reward models, and policy/value predictions without a known environment model.

---

## 🧠 Continual Learning

| Paper | Authors | Year |
|-------|---------|------|
| **TITANS: Learning to Memorize at Test Time** | Behrouz et al. | 2025 |
| **MIRAS (Associative Memory for Sequence Models)** | Behrouz et al. | 2025 |

**TITANS (2025):** Introduces a trainable external memory module that caches and attends to historical context — improves long-term memory in sequence models beyond standard context windows.

**MIRAS (2025):** Unifying framework that recasts Transformers and RNNs as associative memory architectures, with explicit attentional biases and retention gates. Proposes new architectures: Moneta, Yaad, Memora.

---

## 🖥️ Agentic Operating Systems

### Background & Surveys

| Paper | Venue | Year |
|-------|-------|------|
| **LLM as OS, Agents as Apps: Envisioning AIOS, Agents and the AIOS-Agent Ecosystem** | — | — |
| **AIOS: LLM Agent Operating System** (Rutgers) | COLM | 2025 |
| **AgentOS Survey** — MLLM-based agents on real OS | ACL | 2025 |
| **Memory OS of AI Agent** | — | — |
| **MemOS: A Memory OS for AI Systems** | — | — |
| **Towards Agentic OS: An LLM Agent Framework for Linux Schedulers** | — | — |

Papers in this cluster explore LLMs as the "kernel" of an OS, with agents as application-layer processes. Key themes: memory management, scheduling, orchestration, and multi-agent coordination.

---

### 2026 AI-Native OS Stack

| Paper | Key Idea | Ref | Year |
|-------|----------|-----|------|
| **Neural Computers** (Meta AI & KAUST) | Model *is* the computer — latent state replaces the hardware stack | arXiv 2604.06425 | Apr 2026 |
| **AgentOS** (Liu et al.) | GUI replaced by NUI; OS as a continuous intent-mining pipeline | arXiv 2603.08938 | Mar 2026 |
| **Qualixar OS** | Full multi-agent runtime with formal topology semantics | arXiv 2604.06392 | Apr 2026 |

**Neural Computers (Apr 2026)** — *Zhuge, Zhao, Liu et al., Meta AI & KAUST* · `arXiv:2604.06425`
Proposes NCs as a new machine form unifying computation, memory, and I/O inside a **single learned latent state**. Prototypes use video models to roll out terminal and GUI interfaces. Notable results: GUI cursor control at 98.7%, arithmetic-probe accuracy 4% → 83% with reprompting. Long-term target: the *Completely Neural Computer (CNC)* — no separation between software, hardware model, and state.

**AgentOS (Mar 2026)** — `arXiv:2603.08938`
Diagnoses the core problem: agents running on legacy GUIs/CLIs create architectural mismatch, fragmented permissions ("Shadow AI"), and context fragmentation. Proposes a **Natural User Interface (NUI)** kernel that treats OS scheduling as *intent scheduling* — a continuous pipeline of sequential pattern mining, skill retrieval, and evolving personal knowledge graphs.

**Qualixar OS (Apr 2026)** — `arXiv:2604.06392`
Most complete multi-agent runtime specification published so far. Supports 10 LLM providers, 8+ agent frameworks, 7 transports. Features 12 multi-agent topology execution semantics with formal termination conditions, a Q-learning + Bayesian POMDP model router, consensus-based judge pipeline with **Goodhart's Law detection**, and 4-layer content attribution with HMAC signing and steganographic watermarks.

---

## ⚡ Hardware Frontiers

| System | Key Result | Year |
|--------|-----------|------|
| **Intel Hala Point** | 1.15B neurons; 70× faster, 5,600× more energy-efficient than GPU edge AI for CL tasks | 2026 |
| **41M Photonic Neurons on Metasurface Chip** | Matches ResNet/ViT performance; 1,000× lower compute time & energy vs. GPU | arXiv 2504.20416, 2025 |

**Intel Hala Point (2026):** Neuromorphic system at 1.15 billion neurons. Benchmark results show 70× faster performance and 5,600× greater energy efficiency over GPU-based edge AI for continual learning tasks.

**41M Photonic Neurons (2025)** — `arXiv:2504.20416`
Large-scale optical neural network (ONN) on a 10mm² metasurface chip. A single-layer metasurface ONN matches deep model performance while reducing compute time and energy by over 1,000× vs. state-of-the-art GPUs. Leverages photon speed and bandwidth to overcome the memory wall and power wall of von Neumann architectures.

---

## 🗺️ Open Research Gap

The Meta Neural Computers paper explicitly leaves open: **runtime governance, symbolic stability, and controlled reprogramming**. Combined with AgentOS's *intent-as-syscall* model and the photonic hardware substrate, the unoccupied thesis is:

> *"What does a formal theory of scheduling, memory protection, and process isolation look like when the 'CPU' is a latent video model running on a photonic metasurface?"*

No paper answers that yet.

---

*Last updated: May 2026*
