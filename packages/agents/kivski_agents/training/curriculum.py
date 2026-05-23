"""Curriculum-style stage progression for the trainer.

The :class:`CurriculumManager` walks through
:class:`kivski_sim.config.CurriculumStage` entries one stage at a time,
emitting an overridden :class:`KivskiConfig` per stage. The trainer
re-instantiates the vec env (and thus the model, if input dims changed
between stages) whenever :meth:`advance` reports a stage flip.

Stage field semantics:

* ``team_size`` -- overrides ``simulation.team_size``.
* ``max_rounds`` -- overrides ``simulation.max_rounds``.
* ``use_economy`` -- when False, set ``starting_money`` very high so any
  weapon purchase succeeds without having to learn economy management.
* ``episodes`` -- how many episodes (per env) to spend in this stage.

If the curriculum is disabled (``enabled=False`` or empty stages list)
the manager returns the original config unchanged and never advances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kivski_sim.config import CurriculumStage, KivskiConfig


__all__ = ["CurriculumManager", "CurriculumState"]


# An effectively-infinite starting_money value for ``use_economy=False`` stages.
# We don't override the per-weapon costs because the engine treats them as data;
# instead we give the agents enough money that buy decisions are trivially
# affordable. The constant fits in the same ``int`` slot as the standard config.
_NO_ECONOMY_MONEY: int = 12000


@dataclass
class CurriculumState:
    """Snapshot of the curriculum's current position."""

    stage_index: int
    stage_name: str
    episodes_in_stage: int
    total_stages: int
    enabled: bool


class CurriculumManager:
    """Walks through ``cfg.training.curriculum.stages``.

    The manager is *not* responsible for re-building the trainer's runtime
    structures (vec env, model, etc.) -- it only emits new configs and
    flags when a stage flip happens. The trainer reacts by rebuilding
    whatever depends on the changed config fields.
    """

    def __init__(self, base_cfg: KivskiConfig) -> None:
        self._base: KivskiConfig = base_cfg
        self.enabled: bool = bool(base_cfg.training.curriculum.enabled)
        self.stages: list[CurriculumStage] = list(base_cfg.training.curriculum.stages)
        # When curriculum is enabled but the list is empty we fall back to
        # the disabled behaviour so the trainer always sees a usable config.
        if self.enabled and not self.stages:
            self.enabled = False
        self._current_idx: int = 0
        self._episodes_in_stage: int = 0

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> CurriculumStage | None:
        if not self.enabled:
            return None
        if 0 <= self._current_idx < len(self.stages):
            return self.stages[self._current_idx]
        return None

    @property
    def current_stage_name(self) -> str:
        if not self.enabled:
            return "default"
        stage = self.current_stage
        return str(stage.name) if stage is not None else "completed"

    @property
    def state(self) -> CurriculumState:
        return CurriculumState(
            stage_index=int(self._current_idx),
            stage_name=str(self.current_stage_name),
            episodes_in_stage=int(self._episodes_in_stage),
            total_stages=int(len(self.stages)),
            enabled=bool(self.enabled),
        )

    @property
    def finished(self) -> bool:
        """``True`` once we've walked past the last stage (curriculum-enabled only)."""
        if not self.enabled:
            return False
        return self._current_idx >= len(self.stages)

    # ------------------------------------------------------------------
    # Config emission
    # ------------------------------------------------------------------

    def current_config(self) -> KivskiConfig:
        """Return a :class:`KivskiConfig` with stage overrides applied.

        Returns the base config unchanged if the curriculum is disabled or
        if we've walked past the last stage.
        """
        if not self.enabled or self.finished:
            return self._base
        stage = self.current_stage
        if stage is None:
            return self._base
        raw: dict[str, Any] = self._base.model_dump()
        sim = raw.setdefault("simulation", {})
        sim["team_size"] = int(stage.team_size)
        sim["max_rounds"] = int(stage.max_rounds)
        # Don't switch sides mid-training (see trainer rationale): a huge
        # value effectively disables it.
        sim["side_switch_round"] = max(int(stage.max_rounds) + 1, 999)
        if not bool(stage.use_economy):
            sim["starting_money"] = int(_NO_ECONOMY_MONEY)
        return KivskiConfig.model_validate(raw)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def advance(self, episodes_done_in_stage: int) -> bool:
        """Bump the episodes counter and possibly flip to the next stage.

        Args:
            episodes_done_in_stage: How many new episodes to add to the
                current stage's counter.

        Returns:
            ``True`` if a stage flip happened, ``False`` otherwise.
        """
        if not self.enabled or self.finished:
            return False
        self._episodes_in_stage += max(0, int(episodes_done_in_stage))
        stage = self.current_stage
        if stage is None:
            return False
        budget = int(stage.episodes)
        if self._episodes_in_stage >= budget and budget > 0:
            self._current_idx += 1
            self._episodes_in_stage = 0
            return True
        return False

    def reset(self) -> None:
        """Restart the curriculum from stage 0 (useful for resume / tests)."""
        self._current_idx = 0
        self._episodes_in_stage = 0

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "stage_index": int(self._current_idx),
            "episodes_in_stage": int(self._episodes_in_stage),
            "total_stages": int(len(self.stages)),
        }

    def load_state(self, raw: dict[str, Any]) -> None:
        self._current_idx = int(raw.get("stage_index", 0))
        self._episodes_in_stage = int(raw.get("episodes_in_stage", 0))
