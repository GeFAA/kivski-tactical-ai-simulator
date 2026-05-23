# Roadmap

Honest, opinionated list of what we want to do next. Items are grouped by
time horizon, then loosely ranked inside each group. Anything marked
**deferred** is intentionally not a near-term goal.

---

## Short term (next release, < 1 month)

These are the items that should land before V0.2.

### Training loop

- [ ] Finish the `Trainer` class wiring (currently in progress, task #6).
- [ ] Smoke-mode CLI for `kivski-train`: `python -m kivski_agents.training
      smoke` should boot, run 10 episodes on 2 envs, write a checkpoint,
      and exit clean in under a minute. Used in CI as a sanity gate.
- [ ] `--resume <path.pt>` end-to-end test (load checkpoint, continue
      training without RNG drift).

### Tests

- [ ] Bring `tests/smoke/` to green: a single 5v5 match between two
      random baselines plus a single PPO update step.
- [ ] CI matrix expanded to also run `pytest tests/integration -q`.

### Viewer

- [ ] Communication arrow overlay (task #11).
- [ ] Charts panel (winrate / economy / action distribution, task #12).
- [ ] FoV cone overlay polish (currently functional but rough at low
      tile sizes).

### Docs

- [x] README + ARCHITECTURE + ML + MAP_FORMAT + OBSERVATION_ACTION_SPEC.
- [ ] First screenshot of a live match (`docs/screenshots/live-viewer.png`).
- [ ] Short Loom-style video walking through a training run.

---

## Mid term (3 - 6 months)

Things we want, but only after the short-term items are stable.

### Performance

- [ ] Subprocess-based `VecEnv` (currently sync only). Target a 4 - 8x
      throughput improvement on a 16-core machine.
- [ ] Numba inlining of the visibility raycast hot path. Currently
      ~20% of step time on dustline.
- [ ] Optional Rust port of `combat.py` (kept behind a feature flag so
      the pure-Python build still works on weird platforms).

### Game content

- [ ] Dropped-weapon pickup. Currently a kill drops the weapon visually
      but agents cannot acquire it.
- [ ] Grenades (HE), smokes, flashes. Adds a real utility-buy decision
      to the buy phase.
- [ ] Second original map. Goal: an "open" layout (more long sight
      lines, less mid control) to stress-test the policy against
      different visibility distributions.
- [ ] Asymmetric round timers (defenders defuse faster with a kit; kit
      becomes a buyable item).

### RL stack

- [ ] Opponent-modelling head: predict the opponent's policy
      distribution as an auxiliary task. Cheap to add, often improves
      coordinated play.
- [ ] PSRO-style league (Lanctot et al. 2017) replacing the current
      simple snapshot pool. Behind a flag - the snapshot pool stays as
      the default.
- [ ] Per-side asymmetric reward shaping. Defender-side and
      attacker-side have very different objective structures and
      lumping them together is a known weakness.
- [ ] Mixed-precision (bf16) training when on a supported GPU.

### Eval

- [ ] Replay viewer mode in the React app: load a `.replay` file and
      scrub through it offline.
- [ ] Action-distribution heatmap (which head goes where, per round
      phase) in the metrics panel.
- [ ] Time-to-first-contact and trade-fragging-rate aggregates.

---

## Long term (research directions)

Higher-risk items where we are not yet sure the outcome is worth the
cost. Treat each as "we would love a PR / collab".

### Distributed training

- [ ] Ray-based actor / learner split for true multi-machine training.
      Today we are single-machine and gated by Python.

### Bigger team sizes

- [ ] Scaling to 10v10 or 16v16. The centralised critic's joint
      observation width grows linearly with team size; an obvious next
      step is to swap it for an attention-pooling critic that does not
      grow.

### Better comm channels

- [ ] Latent-language comm: instead of `(signature, value)` pairs in
      `R^d`, force the comm channel through a discrete codebook
      (VQ-VAE style). Trades expressivity for interpretability.
- [ ] Bandwidth scheduling: allow agents to "spend" a per-round
      bandwidth budget on the comm channel. The current gate is a soft
      sparsity prior; a hard budget would force a tougher tradeoff.

### Curriculum

- [ ] Procedural map generator. Right now a curriculum advances team
      size and economy; we would also like to advance map difficulty.
- [ ] Reward shaping that anneals not just amplitude but *structure*
      (drop entire shaping terms one at a time as the policy matures).

### Tooling

- [ ] ONNX export pipeline for the policy network so the runtime can be
      embedded outside Python.
- [ ] Hosted demo server with a public replay archive.
- [ ] WebGPU renderer fallback in the viewer for the eventual day
      WebGL gets retired.

---

## Deferred (probably not worth doing)

For completeness, things we have considered and consciously parked:

- **3D engine.** Out of scope; the value of a 3D engine for the
  research questions we care about is small relative to the engineering
  cost.
- **Real-time human-in-the-loop play.** Fun but a serious distraction
  from the training story.
- **Anti-cheat / matchmaking.** Not a multiplayer product.
- **Mobile UI for the viewer.** The viewer is a desktop debugging tool;
  forcing it to be responsive is not worth the design budget.
- **Replacing PettingZoo with a custom env interface.** PettingZoo is
  fine and the integration surface for third-party libraries is
  valuable.

---

## How to propose a roadmap change

Open a GitHub issue with the `roadmap` label. Reasonable proposals look
like:

- "Move X from mid-term to short-term because Y."
- "Add Z to long-term, here is a sketch."
- "Drop Q because it is solved by R."

We are not strict about the structure; please just include enough context
that someone landing on the issue cold can decide whether to dig in.
