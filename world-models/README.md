# World Models — Ha & Schmidhuber (2018)
> *"Can agents learn inside of their own dreams?"*

---

## 1. Context: Why World Models Mattered

**Before World Models (~2018), model-free RL had a hard constraint:**

- Agents had to interact with the real environment **for every learning step**
- Pain points:
  - **Expensive**: Millions of environment steps needed
  - **Policy network/RNN had to stay small** — because of the credit assignment problem (backprop through long time horizons gets messy), and large RNNs = vanishing gradients

---

## 2. The World Models Shift

**Core idea:**
1. First, let the agent build an **internal generative model** of how the environment works
2. Then use that model as a new "dream" environment — the agent can train through **thousands of imagined trajectories**

**Why it's powerful:** Unlike model-free RL, we make the RNN brain **large** since prediction is easy, and the controller stays tiny.

---

## 3. Architecture: V + M + C

```
World Models = V (Vision) + M (Memory) + C (Controller)
```

| Component | Role |
|-----------|------|
| **V — VAE** | Compresses raw frames → 32-dim latent `z` |
| **M — MDN-RNN** | Predicts next `z`, maintains hidden state `h` |
| **C — Controller** | Tiny linear layer: uses `[z, h]` → action |

**Flow at each timestep:**
```
observation → V → z
              [z, h] → C → action
              [z, action] → M → h_next, predicted z_next
```

---

## 4. V — Variational Autoencoder

**Job:** Compress large 3D images (64×64×3) into a 32-dim latent `z` that is both **accurate** and **usable**.

**How it trains:**
1. **Encoder**: Takes original observed frame → compresses to latent `z`
2. **Decoder**: Takes `z` → reconstructs the frame
3. **Loss**: MSE reconstruction loss + KL divergence

**VAE Loss formula:**
```
L = ||x - Decoder(z)||² + Σ(log σ² - μ² - σ²)
```
- **MSE** forces decoder to rebuild image accurately
- **KL** keeps `z` well-behaved so we can sample/predict it later

**The Organized Library Analogy:**
- *Only MSE*: You'd cram each book into its own tiny random corner — perfect recall, but no structure
- *Add KL*: Force all books into one tidy circular room centered at 0. The space is now **semantic** — moving 1 step in any direction gives a slightly different but valid book. That's why the RNN can later learn `z_t+1 ≈ z_t[0...n]`

**Reparameterization Trick:** Encoder outputs `μ, σ`, then `z = μ + σ * ε`, `ε ~ N(0,1)`. Lets gradients flow through the random sampling.

---

## 5. M — MDN-RNN (Memory Model)

**Job:** Predict the future latent `z_{t+1}`, not just one guess — a **mixture of Gaussians**.

**Why MDN?** The world is uncertain/stochastic. M outputs params for a mixture of Gaussians: `P(z_{t+1} | a_t, z_t, h_t, z_{t-1}, h_{t-1}, ...)`

**Why RNN?** Needs memory `h_t` to capture temporal stuff not in current `z_t` — velocity, object permanence, enemy intent.

**How the MDN-RNN is trained — Data Pipeline:**
1. Collect rollouts: Run a random policy in the real env for ~10k episodes. Save sequences: `[(obs_0, a_0, r_0), (obs_1, ...)]`
2. Convert to latents: Use the frozen VAE encoder `z = Encoder(obs)`. Now you have `[(z_0, a_0), (z_1, a_1), (z_2, a_2), ...]`
3. **Supervised training**: This is just next-step prediction. For each `t`:
   - Input: `z_t, a_t, h_t`
   - Target: `z_{t+1}`
   - Loss: `-log P(z_{t+1} | mdn_output)` using the mixture

**Output layer size explained:**
```python
self.mdn = nn.Linear(h_dim, n_mixtures * (2*z_dim + 1))
```
For each of the `K` mixtures and each of the `D` latent dims, we need:
- `K×1` weight per mixture
- `K×D` mean per dim per mixture
- `K×D` std per dim per mixture

**Total: K + K×D + K×D = K×(2D + 1)**. With K=5, D=32: `5×65 = 325` outputs.

---

## 6. C — Controller

**Job:** Tiny linear layer that maps `[z_t, h_t]` → action.

**Why so small?** The heavy lifting (understanding the world) is done by V and M. C just needs to learn *what action to take* given a rich representation.

```python
# The entire controller
action = W @ [z, h] + b
```

Because C is tiny, it can be optimized with **evolution strategies (CMA-ES)** instead of gradient descent — no backprop needed through V or M.

---

## 5. Files in This Directory

| File | Description |
|------|-------------|
| `vae.py` | Variational Autoencoder (V) — simplified core |
| `mdn_rnn.py` | MDN-RNN Memory Model (M) — simplified core |
| `controller.py` | Controller (C) — simplified core |
| `world_model.py` | Full pipeline combining V + M + C |

---

## References

- Ha, D. & Schmidhuber, J. (2018). *World Models*. [https://worldmodels.github.io](https://worldmodels.github.io)
- Paper: [arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)
