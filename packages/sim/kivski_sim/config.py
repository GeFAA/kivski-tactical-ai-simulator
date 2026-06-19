"""Strongly typed config models loaded from YAML.

Use `load_config(path)` to read configs/default.yaml (or any override).
Env vars take final precedence (prefixed with `KIVSKI_`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class SimulationConfig(_Frozen):
    map: str = "dustline"
    team_size: int = 5
    max_rounds: int = 24
    side_switch_round: int = 12
    round_time_seconds: float = 90
    bomb_timer_seconds: float = 40
    plant_time_seconds: float = 3.0
    defuse_time_seconds: float = 5.0
    defuse_time_with_kit_seconds: float = 3.0
    buy_time_seconds: float = 15
    tick_rate_hz: int = 10
    max_ticks_per_round: int = 900
    starting_money: int = 800
    # Number of engine ticks per env.step(). Policy chooses once, the
    # engine runs `frame_skip` inner ticks with the same action, rewards
    # accumulate. 1 = no skip (live viewer); 4 = standard MARL frame-skip
    # used by the training trainer to reduce variance and speed up
    # convergence ~2-3x.
    frame_skip: int = 1


class EconomyConfig(_Frozen):
    reward_round_win: int = 3250
    reward_round_loss_base: int = 1900
    reward_round_loss_increment: int = 500
    reward_round_loss_max: int = 3400
    reward_kill_rifle: int = 300
    reward_kill_pistol: int = 300
    reward_kill_smg: int = 600
    reward_kill_sniper: int = 100
    reward_bomb_plant: int = 300
    reward_bomb_defuse: int = 250
    reward_bomb_detonate: int = 150


class CombatConfig(_Frozen):
    base_accuracy_standing: float = 0.78
    base_accuracy_moving: float = 0.32
    base_accuracy_crouched: float = 0.92
    reaction_time_min_ticks: int = 1
    reaction_time_max_ticks: int = 5
    los_check_step: float = 0.5
    cover_damage_multiplier: float = 0.55


class ObservationConfig(_Frozen):
    teammate_slots: int = 4
    last_known_enemies: int = 5
    sound_event_slots: int = 6
    received_message_slots: int = 5
    history_length: int = 4


class ActionShapeConfig(_Frozen):
    move_intents: int = 9
    micro_actions: int = 6
    comm_actions: int = 9
    buy_options: int = 8


class AgentSubConfig(_Frozen):
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    action: ActionShapeConfig = Field(default_factory=ActionShapeConfig)


class MLConfig(_Frozen):
    algo: str = "mappo"
    hidden_size: int = 256
    gru_layers: int = 1
    comm_attention_heads: int = 4
    comm_embedding_dim: int = 64
    gumbel_temperature: float = 1.0
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    # Sized to match the default 32 envs / 256 rollout steps batch.
    minibatch_size: int = 2048
    learning_rate: float = 3.0e-4
    entropy_coef: float = 0.015
    value_coef: float = 0.5
    gae_lambda: float = 0.95
    gamma: float = 0.99
    max_grad_norm: float = 0.5


class CurriculumStage(_Frozen):
    name: str
    team_size: int
    max_rounds: int
    use_economy: bool
    episodes: int


class CurriculumConfig(_Frozen):
    enabled: bool = False
    stages: list[CurriculumStage] = Field(default_factory=list)


class TrainingConfig(_Frozen):
    # Default tuned for an 8-16 core box. The CLI's ``--auto-envs`` flag
    # bumps this to ``max(8, min(64, cpu_count - 2))`` for the host.
    num_envs: int = 32
    rollout_steps: int = 256
    total_episodes: int = 50000
    checkpoint_every_episodes: int = 500
    eval_every_episodes: int = 250
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)


class LeagueConfig(_Frozen):
    population_size: int = 4
    snapshot_every_episodes: int = 1000
    exploit_fraction: float = 0.25
    random_fraction: float = 0.10
    scripted_fraction: float = 0.10


class RewardShapingConfig(_Frozen):
    enabled: bool = True
    decay_after_episodes: int = 20000
    damage_dealt_per_hp: float = 0.005
    damage_received_per_hp: float = -0.003
    survival_per_second: float = 0.001
    successful_plant: float = 0.5
    successful_defuse: float = 0.4
    bomb_pickup: float = 0.05
    useful_trade: float = 0.15
    pointless_death: float = -0.20
    map_control_per_tile: float = 0.0008
    defenders_elim_bonus: float = (
        0.0  # one-shot terminal bonus per attacker when DEFENDERS_ELIM outcome fires
    )
    plant_progress_per_second: float = (
        0.0  # per-second reward for carrier while actively planting (scaled by tick_dt)
    )


class RewardCurriculumStage(_Frozen):
    """Single stage of the reward curriculum -- which feature buckets are on.

    ``features`` is a list of opt-in reward-shaping buckets. Special token
    ``"all"`` enables every bucket. The trainer advances stages after the
    cumulative episode threshold; ``episodes=-1`` means "forever from now".
    """

    name: str
    episodes: int = 0  # -1 = open-ended (final stage)
    features: list[str] = Field(default_factory=lambda: ["all"])


class RewardCurriculumConfig(_Frozen):
    """Sequential gating of reward-shaping buckets.

    Stage 0 typically only enables ``kill`` + ``survive`` so agents learn
    aggressive combat fast. Later stages layer plant/defuse and finally the
    full set including economy / map control. Disabled by default for
    backward compatibility with v0.2 -- set ``enabled: true`` in the YAML
    (or in code) to opt in.
    """

    enabled: bool = False
    stages: list[RewardCurriculumStage] = Field(default_factory=list)


class TelemetryConfig(_Frozen):
    backend: str = "csv"
    log_every_episodes: int = 10
    flush_every_seconds: int = 5


class EvalScenario(_Frozen):
    name: str
    team_size: int | None = None
    money_override: int | None = None
    attackers: int | None = None
    defenders: int | None = None
    scenario: str | None = None


class EvaluationConfig(_Frozen):
    baselines: list[str] = Field(
        default_factory=lambda: ["random", "scripted_rush", "scripted_hold", "frozen_latest"]
    )
    scenarios: list[EvalScenario] = Field(default_factory=list)
    matches_per_scenario: int = 20


class ServerConfig(_Frozen):
    host: str = "127.0.0.1"
    port: int = 8000
    tick_broadcast_hz: int = 20
    max_clients: int = 4


class KivskiConfig(_Frozen):
    seed: int = 20260523
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    economy: EconomyConfig = Field(default_factory=EconomyConfig)
    combat: CombatConfig = Field(default_factory=CombatConfig)
    agent: AgentSubConfig = Field(default_factory=AgentSubConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    league: LeagueConfig = Field(default_factory=LeagueConfig)
    reward_shaping: RewardShapingConfig = Field(default_factory=RewardShapingConfig)
    reward_curriculum: RewardCurriculumConfig = Field(default_factory=RewardCurriculumConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply `KIVSKI_*` env vars on top of the YAML dict.

    Supported overrides (string-typed and parsed by pydantic):
      KIVSKI_SEED, KIVSKI_NUM_ENVS, KIVSKI_DEVICE (consumed by trainer, not here)
    """
    if (v := os.getenv("KIVSKI_SEED")) is not None:
        raw["seed"] = int(v)
    if (v := os.getenv("KIVSKI_NUM_ENVS")) is not None:
        raw.setdefault("training", {})["num_envs"] = int(v)
    return raw


def load_config(path: str | os.PathLike[str] | None = None) -> KivskiConfig:
    """Load a config YAML, applying env-var overrides."""
    if path is None:
        path = os.getenv("KIVSKI_DEFAULT_CONFIG", "configs/default.yaml")
    p = Path(path)
    if not p.exists():
        # Fall back to defaults if no config file is present (useful for tests).
        return KivskiConfig()
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw = _apply_env_overrides(raw)
    return KivskiConfig.model_validate(raw)
