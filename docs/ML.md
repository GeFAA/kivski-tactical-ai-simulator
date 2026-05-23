# ML stack

This document covers the algorithmic side of Kivski: MAPPO, the recurrent
core, autoregressive action heads, TarMAC-style learned communication,
reward shaping, league play, and the curriculum. For the engine /
observation wire format read [`OBSERVATION_ACTION_SPEC.md`](OBSERVATION_ACTION_SPEC.md);
for module layout read [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 1. The problem in one paragraph

We have a cooperative-competitive team game with partial observability,
sparse outcome rewards, a structured multi-head action space, and N=5
agents per side that must coordinate without any predefined roles. A
trained policy needs to handle map awareness, partial information from
visibility and sound, an economy decision every round, real-time aim and
movement, *and* limited-bandwidth communication with teammates. That is too
much for a single-agent algorithm to learn from scratch, so we use **MAPPO
with parameter sharing + a centralised critic + recurrent actors + a
learned attention-based communication channel**.

---

## 2. PPO refresher

Proximal Policy Optimisation (Schulman et al., 2017) is the workhorse here.
The clipped surrogate objective:

```
L^CLIP(theta) = E_t [ min( r_t(theta) * A_hat_t,
                            clip(r_t(theta), 1 - eps, 1 + eps) * A_hat_t ) ]
```

with

```
r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
```

plus a value-function loss and an entropy bonus:

```
L = -L^CLIP + c_v * (V_phi(s_t) - V_target_t)^2 - c_e * H(pi_theta)
```

`V_target_t` is computed with Generalised Advantage Estimation (GAE,
Schulman et al. 2015) using `gamma=0.99` and `gae_lambda=0.95` by default.

Hyperparameters live in `MLConfig` and are surfaced in
`configs/default.yaml`:

| Field             | Default | What it does |
|-------------------|---------|--------------|
| `ppo_clip`        | 0.2     | The `eps` in the clip. |
| `ppo_epochs`      | 4       | Passes over each rollout. |
| `minibatch_size`  | 1024    | Inner shuffle for SGD. |
| `learning_rate`   | 3e-4    | Adam step size. |
| `entropy_coef`    | 0.015   | Multiplier on `-H(pi)`. |
| `value_coef`      | 0.5     | Multiplier on the value MSE. |
| `gae_lambda`      | 0.95    | GAE smoothing. |
| `gamma`           | 0.99    | Discount factor. |
| `max_grad_norm`   | 0.5     | Global gradient clip. |

---

## 3. From PPO to MAPPO

PPO is single-agent. For multi-agent cooperative play we use **MAPPO**
(Yu et al., NeurIPS 2022). Two practical changes:

1. **Parameter sharing across agents on the same team.** A single actor
   network is used by every agent on the team. Agents are not identical -
   they differ in observation (egocentric) and recurrent hidden state -
   but they all share weights. Sample efficiency goes up linearly with
   team size and the policy generalises across positions.

2. **Centralised critic.** The value function `V_phi` takes the *joint*
   observation (concatenation of every teammate's observation, plus
   optional global features) rather than a single agent's view. This
   resolves the credit-assignment problem that pure independent PPO has
   in team games. At inference time the critic is unused, so this does
   not violate the decentralised-execution property.

`kivski_agents.factory.infer_joint_obs_dim(obs_dim, team_size,
global_features)` is the helper that computes the critic's input width.

Yu et al. report that MAPPO with sensible tricks (advantage normalisation,
shared parameters, separate value clip) matches or beats QMIX, MADDPG, and
IPPO on a wide range of cooperative benchmarks. We follow their tricks
list.

---

## 4. The actor-critic network

Top-level: `kivski_agents.networks.KivskiActorCritic`.

```
            +------------------------+
obs[t] ---> |  ObservationEncoder    |  (MLP: obs_dim -> 256)
            +-----------+------------+
                        |
                        v
            +------------------------+
            |  RecurrentCore (GRU)   |  (256 -> 256)
            +-----------+------------+
                        |
              +---------+---------+
              v                   v
   +-----------------+   +-----------------+
   |  CommEncoder    |   |  ActorHeads     |
   |  -> (sig, val)  |   |  (auto-regr.)   |
   +--------+--------+   +-----------------+
            |
            v
   +-----------------+
   |  CommAttention  |   reads sig+val from teammates
   +--------+--------+
            |
            v
   +-----------------+
   |  CommGate       |   Gumbel-Sigmoid, sparse broadcast
   +-----------------+


joint_obs[t]  -->  ValueHead  -->  V_phi(s_t)
```

### 4.1 `ObservationEncoder`

A plain 2-layer MLP, ReLU activations, `obs_dim -> ml.hidden_size`. The
exact `obs_dim` is computed by `kivski_sim.obs_decoder.get_observation_dim`
from the current config.

### 4.2 `RecurrentCore`

A GRU with `ml.gru_layers` layers (default 1) and width
`ml.hidden_size` (default 256). We use a GRU rather than an LSTM because:

- GRUs have fewer parameters and train slightly faster.
- We do not need a separate cell state for this horizon (~256 ticks).
- They are easier to mask correctly across episode boundaries in the
  rollout buffer.

We mask hidden states at episode boundaries inside `RolloutBuffer` so the
recurrent state never bleeds between rounds.

### 4.3 `ActorHeads` (autoregressive)

Five heads matching the action-space layout:

```
move_intent  (9 logits)
micro        (6 logits)
comm         (9 logits)
buy          (8 logits)
aim_target   (2 * team_size + 1 logits)
```

Heads are sampled **autoregressively**: after `move_intent` is sampled it
is embedded and concatenated to the GRU output before the next head's
logits are computed, and so on. This lets later heads condition on earlier
choices - e.g. `aim_target` knows whether the agent decided to `SPRINT` or
`CROUCH_HOLD`.

The autoregressive ordering used in code is `move -> micro -> comm -> buy
-> aim_target`. Aim is last so it can condition on the chosen micro
posture and movement.

Log-probabilities and entropies are summed across heads when computing the
PPO loss.

### 4.4 `ValueHead` (centralised critic)

A 2-layer MLP that maps the joint observation to a scalar value. Inputs:

```
joint_obs = concat([
    obs_for(agent_i) for i in range(team_size)
])
```

The critic's output is shared across all of a team's agents (they share a
value target by construction since they share the reward in MAPPO with
team rewards).

### 4.5 Initialisation and tricks

Following Yu et al. and Engstrom et al. ("Implementation Matters"):

- Orthogonal init on all linear layers, gain `sqrt(2)` on hidden layers,
  gain `0.01` on actor head outputs, gain `1.0` on the value head output.
- Bias init `0`.
- Layer norm on the recurrent core input.
- Advantage normalisation per minibatch.
- Value loss clipping (mirrors the policy clip).

---

## 5. TarMAC: targeted multi-agent communication

We borrow the recipe from Das et al. (ICML 2019).

### 5.1 Why a learned channel

Hard-coded ping systems pre-decide the vocabulary of communication. A
learned channel lets the team discover whatever signals are actually
predictive of round outcomes. That can be "enemy spotted" but also things
like "I'm holding angle X so don't push it" that are difficult to
enumerate by hand.

### 5.2 The channel

Each agent's `CommEncoder` produces

```
sig_i  in R^{comm_embedding_dim}        (signature, "who should care")
val_i  in R^{comm_embedding_dim}        (value, "what I want to say")
```

A multi-head attention computes each receiver's read:

```
weights_{i,j} = softmax_j ( sig_i^T q_j )         (per attention head)
read_i        = sum_j ( weights_{i,j} * val_j )
```

`q_j` is a learned query projection of receiver `j`'s GRU state. We use
`ml.comm_attention_heads = 4` heads, each of width
`comm_embedding_dim / num_heads`.

The receiver's `read_i` vector is concatenated to its observation
embedding before the next tick's forward pass, so the receiver sees a
condensed summary of everything its teammates broadcast.

### 5.3 The Gumbel-Sigmoid gate

Broadcasting on every tick is wasteful and tends to collapse the channel
into a constant background signal. We add a `CommGate` head that produces
a per-agent Bernoulli `p_i` and samples a broadcast decision through a
Gumbel-Sigmoid relaxation:

```
g_i = sigmoid( (logit_i + Gumbel_noise) / tau )
      with tau = ml.gumbel_temperature  (default 1.0)
```

We multiply the broadcast `val_i` by `g_i`. Forward path is the relaxed
gate; backward path is straight-through (`g_i.detach() + p_i - p_i.detach()`
trick).

A small auxiliary loss

```
L_gate = beta * mean( g_i )
```

penalises always-on gates, encouraging sparsity. `beta` defaults to a
small constant (0.01); raise it if the channel saturates.

### 5.4 The discrete `CommAction` head

The `comm` actor head picks one of 9 *categories* (no-comm + 8 callouts
like `PING_LOCATION`, `WARN_DANGER`, ...). These categories are
*human-readable labels* that the viewer renders for debugging. The actual
payload routed through TarMAC is the learned `val_i` vector, gated by
`g_i`.

The categories are *not* hardcoded to specific behaviours - the policy is
free to learn that, say, `WARN_DANGER` always pairs with a particular
payload, or that two different categories share semantics. They give us a
discrete labelling that is easy for a human to grep through replays and
hard for the agent to over-specialise on.

---

## 6. Reward design

Reward at time `t` for agent `i` on team `T`:

```
r_t^i = r_outcome_t^T + alpha(epoch) * r_shape_t^i
```

where `r_outcome` is the sparse team reward (round win / loss, plant /
defuse), `r_shape` is the dense agent-local shaping signal, and `alpha`
linearly decays from `1` to `0` over
`reward_shaping.decay_after_episodes` (default 20 000):

```
alpha(epoch) = max(0, 1 - epoch / reward_shaping.decay_after_episodes)
```

### 6.1 Components

| Name | Tier | Per-event reward | Why |
|------|------|------------------|-----|
| Round win | Outcome | +1.0 (team) | The only signal we trust long-term. |
| Round loss | Outcome | -1.0 (team) | Symmetric, balanced. |
| Plant | Shaping | +0.5 (team) | Aligns attackers with the side-specific objective. |
| Defuse | Shaping | +0.4 (team) | Same on the defender side. |
| Damage dealt per HP | Shaping | +0.005 (per-agent) | Reduces variance early; tied to whoever fired. |
| Damage received per HP | Shaping | -0.003 (per-agent) | Discourages free-frags against you. |
| Survival per second | Shaping | +0.001 (per-agent) | Mild bias against suicidal pushes. |
| Bomb pickup | Shaping | +0.05 (per-agent) | Gets the bomb out of spawn. |
| Useful trade (kill within X ticks of teammate death) | Shaping | +0.15 (per-agent) | Encourages trade fragging. |
| Pointless death (died without dealing damage / providing info) | Shaping | -0.20 (per-agent) | Penalises low-info deaths. |
| Map control per tile | Shaping | +0.0008 (team) | Mild incentive to expand territory. |

### 6.2 Why the decay schedule

Shaping is a double-edged sword: it gets a policy off the ground but locks
in heuristics that may diverge from the true objective once the policy is
strong. We follow the standard "shape early, fade late" recipe:

- For the first `decay_after_episodes` episodes the policy uses outcome +
  shaping.
- The shaping coefficient `alpha` linearly anneals to zero, so the policy
  is gradually forced to keep the same behaviour without the dense crutch.
- Past the decay horizon the policy is being graded purely on outcomes,
  which is what we ultimately care about.

The decay horizon is the most fragile knob in this config; if you change
it, validate against the eval suite.

---

## 7. League play

Self-play alone has two well-known failure modes:

1. **Rock-Paper-Scissors loops.** Strategy A beats B beats C beats A. The
   policy oscillates between them without monotonic improvement.
2. **Co-adaptation to your own quirks.** The policy specialises against
   the current self and forgets how to beat naive opponents.

We mitigate with a small **opponent league**.

### 7.1 The pool

`LeagueManager` holds:

- The `random` baseline.
- The two `scripted_*` baselines.
- A FIFO ring of **frozen snapshots** of past selves. Capacity is
  `league.population_size` (default 4). A new snapshot is taken every
  `league.snapshot_every_episodes` (default 1000).

### 7.2 The sampler

Each episode, `OpponentSampler` picks an opponent for the current policy:

```
P(random)   = league.random_fraction      (default 0.10)
P(scripted) = league.scripted_fraction    (default 0.10)
P(exploit)  = league.exploit_fraction     (default 0.25)
P(self)     = 1 - sum of the above        (default 0.55)
```

Within the exploit category we weight snapshots towards the ones the
current self loses against most (a soft PSRO heuristic).

### 7.3 Rating

`EloTracker` (always available) and `TrueSkillTracker` (requires the
optional `trueskill` package) consume match results and update ratings
online. Ratings are surfaced in the telemetry log and in the
`/api/training/runs` endpoint, mostly so a human can sanity-check that
the policy is actually improving.

---

## 8. Curriculum

Optional, disabled by default. When `training.curriculum.enabled: true`:

| Stage          | Team | Rounds | Economy | Episodes |
|----------------|------|--------|---------|----------|
| `1v1_no_eco`   | 1    | 6      | off     | 2 000    |
| `2v2_no_eco`   | 2    | 8      | off     | 3 000    |
| `3v3_basic_eco`| 3    | 12     | on      | 5 000    |
| `5v5_full`     | 5    | 24     | on      | 40 000   |

`CurriculumManager.current_stage_overrides()` returns a dict of config
patches that the trainer applies when rebuilding the env on stage
transition. The model is **preserved across stages** - we re-use the GRU
weights, the encoder, the comm channel, and the heads. Only the
`aim_target` head changes width (it depends on team size); we re-init
just that head when the team size changes.

When to enable: if the policy never gets off the ground in pure 5v5
self-play, or if you want a faster path to a recognisable baseline. We
have run the default config from scratch to a workable 5v5 policy without
the curriculum, but it took longer.

---

## 9. Hyperparameter notes

The defaults in `configs/default.yaml` are tuned for a developer laptop
running CPU-only training. A few specific calls:

- **`learning_rate: 3e-4`** - the canonical Adam-PPO value. We have not
  seen wins from cosine schedules at our scale.
- **`entropy_coef: 0.015`** - higher than typical PPO defaults (1e-3 -
  1e-2) because the multi-head action space makes exploration brittle. If
  you see the policy collapsing to "always SPRINT" or "always
  WARN_DANGER", raise this to 0.02 - 0.03.
- **`ppo_epochs: 4`, `minibatch_size: 1024`** - keep wall-clock per update
  short so the league sees more variety.
- **`gae_lambda: 0.95`, `gamma: 0.99`** - standard MAPPO. We have not
  found gains from tuning either.
- **`max_grad_norm: 0.5`** - aggressive clip. The comm channel + GRU
  combine occasionally produce big spikes early in training; this clip
  keeps Adam stable.
- **`comm_attention_heads: 4`, `comm_embedding_dim: 64`** - the smallest
  pair we found that consistently routes information. Halving either kills
  performance; doubling them is a wash.
- **`gumbel_temperature: 1.0`** - the relaxation temperature for the comm
  gate. Cooler (0.5) produces a harder gate but worse gradients; warmer
  (2.0) produces a soft gate that rarely truly cuts off. 1.0 is the
  default in most TarMAC followups.

### Suggested first sweep

If you have budget for a small sweep, the highest-impact knobs are:

1. `entropy_coef` in `{0.005, 0.015, 0.03}`.
2. `reward_shaping.decay_after_episodes` in `{10000, 20000, 40000}`.
3. `league.exploit_fraction` in `{0.10, 0.25, 0.50}`.

That is ~27 runs at the default 50k episodes.

---

## 10. Practical tips for getting it to learn

- **Always sanity-check against `random` first.** If you cannot beat
  random by episode 1000 something is structurally broken.
- **Disable shaping for one diagnostic run.** If shaping is the only
  thing producing apparent progress, the policy will plateau hard once
  the decay kicks in.
- **Watch the comm-gate sparsity.** If `mean(g_i) > 0.8` at convergence
  the gate is saturated and the channel is uninformative. Raise the
  gate-penalty `beta`.
- **Watch the action entropy per head.** Move-intent entropy should drop
  below 0.5 nats once the policy converges; if a head still has near-max
  entropy at episode 30k the heuristic for that head's reward shaping
  may be flat.
- **Use the eval suite, not just training reward.** Training reward is
  contaminated by shaping. The eval scenarios use only the outcome
  reward.

---

## 11. References

- Schulman, Wolski, Dhariwal, Radford, Klimov.
  *Proximal Policy Optimization Algorithms.* 2017.
- Schulman, Moritz, Levine, Jordan, Abbeel.
  *High-Dimensional Continuous Control Using Generalized Advantage
  Estimation.* ICLR 2016.
- Yu, Velu, Vinitsky, Wang, Bayen, Wu.
  *The Surprising Effectiveness of PPO in Cooperative Multi-Agent
  Games.* NeurIPS 2022.
- Das, Gervet, Romoff, Batra, Parikh, Pineau, Rabbat.
  *TarMAC: Targeted Multi-Agent Communication.* ICML 2019.
- Jang, Gu, Poole.
  *Categorical Reparameterization with Gumbel-Softmax.* ICLR 2017.
- Terry et al.
  *PettingZoo: Gym for Multi-Agent Reinforcement Learning.*
  NeurIPS 2021.
- Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph, Madry.
  *Implementation Matters in Deep Policy Gradients: A Case Study on PPO
  and TRPO.* ICLR 2020.
- Heinrich, Silver.
  *Deep Reinforcement Learning from Self-Play in Imperfect-Information
  Games.* 2016.
- Lanctot et al.
  *A Unified Game-Theoretic Approach to Multiagent Reinforcement
  Learning.* (PSRO.) NeurIPS 2017.
