# Track 1 Hackathon Strategy: "Can It Play DOOM?"
**Architecture:** Telemetry-based World Model (RSSM) + Spatial Memory + Privileged Distillation
**Target:** Ace the 4-Tier Zero-Shot Evaluation within a 2GB VRAM / 1 CPU / 40ms step constraint.

---

## 1. The Core Bet: Why We Are Doing This
Standard RL (PPO) fails because it memorizes maps. Pixel-based World Models (DreamerV3) fail because they waste compute rendering wall textures and will blow past our 2GB VRAM / 40ms inference budget. 

Instead, we are building a **Telemetry-Based World Model**. By feeding the official `doom_env` spatial signals (raycasts, entity vectors, vitals) into a Recurrent State-Space Model (RSSM), we can train an agent that actually understands Doom physics, object permanence, and combat dynamics.

---

## 2. System Architecture

### A. Inputs (Ego-Centric Only)
We completely ignore the raw RGB frame to save compute. We only use:
*   **64 Ego-centric Raycasts:** 1D convolutions or circular attention over hit types (wall, door, enemy, etc.) and distances.
*   **Entity Vectors:** A small Set Transformer or Cross-Attention module to handle variable numbers of enemies in sight.
*   **Vitals:** Health, ammo, keycards, velocity.

### B. The World Model (RSSM Core)
*   **Mechanism:** An encoder fuses the inputs into a latent state. A GRU (or Mamba) updates this state over time.
*   **Forward Prediction:** The model predicts the *next* step's raycasts, entity positions, and rewards. 
*   **Curiosity (Tier 2 Driver):** We use the world model's prediction error (surprise) as an intrinsic reward. When the agent sees a new room, prediction error spikes, driving it to explore rather than camp.

### C. Differentiable Spatial Memory (Tier 3 Driver)
*   **The Problem:** The RSSM will forget where a locked door was after 2,000+ ticks of searching for the keycard.
*   **The Fix:** We maintain an egocentric grid matrix in hidden memory. As the agent moves (velocity/odometry), we shift this grid. We use neural read/write heads to "pin" keycards and locked doors to this grid, allowing the agent to backtrack.

### D. The Policy (Actor-Critic)
*   Acts on the concatenated vector of: `[RSSM Latent State, Spatial Memory Readout, Current Vitals]`.

---

## 3. The Training Pipeline (Cloud GPUs Allowed)

To prevent hallucination and accelerate learning, our training loop needs two major additions:

1.  **Procedural WAD Generation (Adversarial Curriculum):**
    *   We cannot train on just one map. We will use a procedural generator (like Oblige/Obsidian or a custom Python script) to generate thousands of random maps.
    *   *Constraint:* Must use Freedoom 2 IWAD rules and `skill 3` difficulty so it transfers zero-shot to the hidden evaluation.
2.  **Privileged Teacher Distillation (The Hack):**
    *   Train a "God-mode" Teacher model that gets internal engine coordinates and map layouts (legal during training).
    *   Train our "Blind" Student (the actual `DoomAgent`) to mimic the Teacher's actions via KL-Divergence loss, while only letting the Student see the ego-centric raycasts. This bypasses the massive exploration bottleneck.

---

## 4. Inference & Sandbox Compliance
When we submit `submission/agent.py`, the code must run in an offline sandbox (RTX 3090, 2GB VRAM cap, 1 CPU thread, ~40ms per step). 

*   **No Latent Rollouts at Test Time:** We *only* run the Encoder -> Posterior Update -> Actor Head. We drop the decoder and imagination rollouts entirely during evaluation to ensure our inference is ~2-5ms, well under the 40ms budget.
*   **Stateless Resets:** The `reset(self)` function must completely wipe the GRU hidden states and the Spatial Memory grid to ensure cross-map contamination doesn't occur.
*   **Load Fast:** Do a dummy tensor forward pass in `__init__(self)` so PyTorch CUDA initialization happens during the generous 60s startup timeout, not the 10s first-action timeout.

---

## 5. Implementation Phases & Task Split

**Phase 1: Environment & Wrappers**
*   Parse the `doom_env` dictionary into clean PyTorch tensors.
*   Setup the procedural WAD generator pipeline (hook it into Gym resets).

**Phase 2: The Core Network**
*   Build the 1D Raycast Encoder and Entity Set-Transformer.
*   Build the core RSSM (GRU/RNN) and the next-step prediction heads.

**Phase 3: Spatial Memory & Curiosity**
*   Implement the velocity-based grid shift for spatial memory.
*   Implement the intrinsic curiosity reward based on next-step prediction loss.

**Phase 4: RL Loop & Distillation**
*   Set up the Teacher model (with privileged inputs) and the Student model.
*   Write the joint loss function (PPO + KL-Distillation).
*   Test on a basic Tier 1 combat map before scaling to procedural multi-room WADs.