"""Top-level training orchestrator.

The :class:`Trainer` owns the full MAPPO training loop:

* curriculum -> active :class:`KivskiConfig`
* vec env -> N parallel :class:`KivskiParallelEnv`
* model + :class:`MAPPOTrainer` for the gradient updates
* :class:`PolicyRunner` for the training-side action sampling
* :class:`LeagueManager` for self-play / baseline / frozen sparring
* :class:`RolloutBuffer` for storing transitions

Per training iteration we (a) sample an opponent from the league, (b)
collect ``rollout_steps`` transitions per env, (c) compute GAE-Lambda
advantages, (d) run the PPO update, (e) log metrics, (f) periodically
checkpoint and snapshot.

Side-switching during training is disabled (the engine still flips
sides for eval): we want the training-side mapping to stay stable so
the learner doesn't suddenly find itself controlling a different team
mid-episode. The trainer forces ``side_switch_round`` to a huge value
on the config it passes to the vec env.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from kivski_sim.config import KivskiConfig

from kivski_agents.buffer import RolloutBuffer
from kivski_agents.eval.runner import EvalResult, EvalRunner
from kivski_agents.eval.scenarios import ALL_SCENARIOS, ScenarioSpec
from kivski_agents.factory import build_model, build_trainer, default_action_dims, infer_joint_obs_dim
from kivski_agents.mappo import MAPPOLoss, MAPPOTrainer
from kivski_agents.metrics import (
    CommUsageStats,
    EpisodeStats,
    TrainStepMetrics,
    comm_usage_to_dict,
    episode_stats_to_dict,
    train_metrics_to_dict,
)
from kivski_agents.policy_runner import PolicyBundle, PolicyRunner
from kivski_agents.telemetry import NoOpSink, TelemetrySink
from kivski_agents.training.curriculum import CurriculumManager, RewardCurriculumManager
from kivski_agents.training.league import LeagueManager
from kivski_agents.training.parallel_vec_env import (
    SubprocVecEnv,
    ThreadedVecEnv,
    make_vec_env,
)
from kivski_agents.training.rollout_collector import RolloutCollector
from kivski_agents.training.vec_env import VecEnvWrapper

# Type alias covering every vec-env backend we ship.
AnyVecEnv = VecEnvWrapper | ThreadedVecEnv | SubprocVecEnv

__all__ = ["TrainerConfig", "Trainer"]


# ---------------------------------------------------------------------------
# Runtime settings
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Runtime settings the trainer needs in addition to :class:`KivskiConfig`."""

    total_episodes: int
    rollout_steps: int
    num_envs: int
    checkpoint_every: int
    eval_every: int
    snapshot_every: int
    log_dir: Path
    checkpoint_dir: Path
    device: torch.device
    map_name: str = "dustline"
    resume_from: Path | None = None
    run_name: str | None = None
    # Eval-time defaults (kept here so the trainer doesn't need a second cfg).
    eval_scenario: str = "default_5v5"
    eval_matches: int = 5
    # How often to print a console summary in addition to telemetry.
    print_every: int = 1
    # Vectorised env backend: "sync" | "thread" | "subproc". The sync backend
    # is the historical default and the most predictable for debugging; the
    # parallel backends gain real throughput once num_envs > a few.
    vec_kind: str = "sync"
    # Number of worker threads/processes for the parallel backends. None ->
    # detected from CPU count at construction time.
    num_workers: int | None = None
    # Auto-save retention: how many *recent* per-episode checkpoints to keep on
    # disk inside ``checkpoint_dir``. Older ones (besides ``best.pt``) are
    # pruned together with their .json sidecars after every save. ``0`` or
    # a negative value disables retention pruning (keep everything).
    retention_count: int = 10


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Drive the MAPPO + league + curriculum training loop."""

    def __init__(
        self,
        cfg: KivskiConfig,
        tcfg: TrainerConfig,
        telemetry: TelemetrySink | None = None,
    ) -> None:
        self.base_cfg: KivskiConfig = cfg
        self.tcfg: TrainerConfig = tcfg
        self.telemetry: TelemetrySink = telemetry if telemetry is not None else NoOpSink()
        self.device: torch.device = torch.device(tcfg.device)
        # Counters.
        self.episode_count: int = 0
        self.update_step: int = 0
        self.env_steps: int = 0

        # 1) Curriculum stage -> effective config.
        self.curriculum: CurriculumManager = CurriculumManager(cfg)
        self.active_cfg: KivskiConfig = self._cfg_with_safety_overrides(self.curriculum.current_config())
        # Reward curriculum (independent of structural curriculum). When
        # enabled, gates the dense-reward shaping buckets per stage so the
        # agent can be guided through "killshoot -> objective -> full".
        self.reward_curriculum: RewardCurriculumManager = RewardCurriculumManager(self.active_cfg)

        # 2) Vec env. ``make_vec_env`` honours ``tcfg.vec_kind`` and falls
        # back to the synchronous wrapper if the parallel backend can't
        # start (e.g. on a Windows host where ``spawn`` failed to pickle
        # something).
        self.vec_env: AnyVecEnv = make_vec_env(
            num_envs=int(tcfg.num_envs),
            cfg=self.active_cfg,
            map_name=tcfg.map_name,
            base_seed=int(self.active_cfg.seed) + 1_000_000,
            kind=str(tcfg.vec_kind),
            num_workers=tcfg.num_workers,
        )
        # Apply the initial reward-curriculum stage to every env in the batch.
        self._apply_reward_curriculum()
        # 3) Model + trainer.
        team_size = int(self.active_cfg.simulation.team_size)
        obs_dim = int(self.vec_env.obs_dim)
        action_dims = default_action_dims(team_size)
        joint_obs_dim = infer_joint_obs_dim(obs_dim, team_size)
        self.model = build_model(
            cfg=self.active_cfg,
            obs_dim=obs_dim,
            joint_obs_dim=joint_obs_dim,
            action_dims=action_dims,
            device=self.device,
        )
        self.mappo: MAPPOTrainer = build_trainer(self.model, self.active_cfg, device=self.device)

        # 4) Training-side runner. We rebuild this per rollout (because the
        # RolloutCollector also wants a fresh runner state) but keep one cached
        # so the trainer-level API can use it for evals too.
        self.training_runner: PolicyRunner = PolicyRunner(
            model=self.model, device=self.device, deterministic=False
        )

        # 5) Buffer.
        self.buffer: RolloutBuffer = RolloutBuffer(
            T=int(tcfg.rollout_steps),
            N_envs=int(tcfg.num_envs),
            n_agents=team_size,
            obs_dim=obs_dim,
            joint_obs_dim=joint_obs_dim,
            n_heads=int(self.vec_env.n_heads),
            hidden_size=int(self.model.hidden_size),
            comm_value_dim=int(self.model.comm_value_dim),
            device=self.device,
            n_teammates=team_size,
            gru_layers=int(self.model.gru_layers),
        )

        # 6) League.
        self.league: LeagueManager = LeagueManager(
            log_dir=Path(tcfg.log_dir),
            cfg=self.active_cfg.league,
            env=self.vec_env.envs[0],
            map_data=self.vec_env.map_data,
            device=self.device,
            main_model=self.model,
        )

        # 7) Optional resume.
        if tcfg.resume_from is not None:
            self.load_checkpoint(Path(tcfg.resume_from))

        # 8) Telemetry hyperparams.
        # Telemetry hiccups must not block training.
        with contextlib.suppress(Exception):
            self.telemetry.log_hyperparams(self._hyperparam_dump())

        # Misc: an internal RNG for opponent sampling.
        self._rng: np.random.Generator = np.random.default_rng(int(self.active_cfg.seed) + 31337)

        # Best-checkpoint tracking. We persist the best score under
        # ``<root>/checkpoints/best.json`` so it survives API restarts and
        # bench runs that reuse the same checkpoint directory tree. The
        # actual weights file lives at ``<root>/checkpoints/best.pt`` (copied
        # from whichever per-run checkpoint produced the best score).
        # ``_best_score`` is loaded eagerly from disk so a long-running
        # training session that resumed after a crash doesn't immediately
        # overwrite a better-scoring previous best.
        self._best_score: float = float("-inf")
        self._best_episode: int = -1
        # Most-recent per-episode checkpoint path, populated lazily by
        # ``save_checkpoint``. Used by ``_maybe_promote_best`` to copy the
        # right .pt onto ``best.pt`` after an eval improves the score.
        self._latest_per_episode_ckpt: Path | None = None
        self._load_best_score_from_disk()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the training loop until ``total_episodes`` is reached."""
        total_target = int(self.tcfg.total_episodes)
        last_eval_at = -1
        last_ckpt_at = -1
        last_snapshot_at = -1
        last_print_at = -1
        # Persist the recurrent hidden state across collector instances so a
        # match that spans multiple rollouts keeps a coherent GRU history.
        carry_hidden: torch.Tensor | None = None

        while self.episode_count < total_target:
            # ---- a) Sample opponent + (re-)build collector --------------
            opponent = self.league.sample_opponent(self._rng)
            collector = RolloutCollector(
                vec_env=self.vec_env,
                training_runner=self.training_runner,
                opponent_sampler=opponent,
                buffer=self.buffer,
                cfg=self.active_cfg,
                device=self.device,
                initial_hidden=carry_hidden,
            )

            # ---- b) Roll out + (c) compute advantages -------------------
            result = collector.collect(int(self.tcfg.rollout_steps))
            carry_hidden = collector.batched_hidden.detach().clone()
            self.env_steps += int(result.total_env_steps)
            self.buffer.compute_advantages(
                last_value=result.last_value,
                gamma=float(self.active_cfg.ml.gamma),
                gae_lambda=float(self.active_cfg.ml.gae_lambda),
            )

            # ---- d) PPO update -----------------------------------------
            loss = self.mappo.update(self.buffer)
            self.update_step += 1

            # ---- e) Process per-episode metrics + Elo ------------------
            self._process_episode_results(
                episode_stats=result.episode_stats,
                opponent_name=opponent.name,
            )
            new_episodes = len(result.episode_stats)
            self.episode_count += new_episodes

            # ---- f) Log to telemetry -----------------------------------
            self._log_step(loss=loss, comm_usage=result.comm_usage, fps=result.fps)

            # ---- g) Optional snapshot + checkpoint + eval --------------
            if self._should_trigger(self.tcfg.snapshot_every, last_snapshot_at):
                snap_path = self._snapshot_main()
                if snap_path is not None:
                    self.league.add_snapshot(snap_path, episode=self.episode_count)
                self.league.save_state()
                last_snapshot_at = self.episode_count

            if self._should_trigger(self.tcfg.checkpoint_every, last_ckpt_at):
                ckpt = self.save_checkpoint(
                    episode=self.episode_count,
                    metadata={
                        "update_step": int(self.update_step),
                        "env_steps": int(self.env_steps),
                        "opponent": str(opponent.name),
                    },
                )
                self.telemetry.log_text("checkpoint", str(ckpt), step=self.update_step)
                last_ckpt_at = self.episode_count

            if self._should_trigger(self.tcfg.eval_every, last_eval_at):
                try:
                    eval_results = self.evaluate()
                    self._log_eval_results(eval_results)
                    # Promote the most-recent per-episode checkpoint to
                    # ``best.pt`` when the combined eval score improves.
                    # We do this *after* evaluate so the score reflects the
                    # weights actually on disk (the per-episode .pt that the
                    # checkpoint_every branch just wrote, or the previous one
                    # when checkpoint_every and eval_every are unaligned).
                    with contextlib.suppress(Exception):
                        self._maybe_promote_best(eval_results)
                except Exception as exc:  # noqa: BLE001 - never fatal
                    self.telemetry.log_text("eval_error", repr(exc), step=self.update_step)
                last_eval_at = self.episode_count

            if self._should_trigger(self.tcfg.print_every, last_print_at):
                self._print_progress(loss=loss, opponent_name=opponent.name, fps=result.fps)
                last_print_at = self.episode_count

            # ---- h) Advance curriculum ---------------------------------
            if self.curriculum.advance(new_episodes):
                self._handle_stage_flip()
                # Hidden state is bound to the old model dimensions; drop it.
                carry_hidden = None

            # ---- h2) Advance reward curriculum -------------------------
            # Reward stages don't change tensor shapes so we never need to
            # drop the hidden state -- we just rebroadcast the feature gate.
            if self.reward_curriculum.advance(new_episodes):
                self._apply_reward_curriculum()

            # If no new episodes were finished in the rollout (very short
            # rollouts on long matches), we should still bump the loop counter
            # to avoid infinite spinning when total_target is small.
            if new_episodes == 0:
                self.episode_count += 0  # explicit no-op for clarity

        # Final flush.
        with contextlib.suppress(Exception):
            self.telemetry.flush()

    # ------------------------------------------------------------------
    # Evaluation hook
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, EvalResult]:
        """Run the current main policy vs each baseline on a chosen scenario.

        Returns a mapping ``baseline_name -> EvalResult``. Updates league
        Elo for each pairing.
        """
        spec = self._resolve_eval_scenario(self.tcfg.eval_scenario)
        runner = EvalRunner(
            scenario=spec,
            cfg=self.active_cfg,
            map_name=self.tcfg.map_name,
            map_data=self.vec_env.map_data,
        )
        # YELLOW = main, BLUE = baseline.
        from kivski_agents.training.league import MainSelfPlayPolicy  # local

        main = MainSelfPlayPolicy(model=self.model, device=self.device)

        results: dict[str, EvalResult] = {}
        baseline_names = ["random", "scripted_rush", "scripted_hold"]
        for bname in baseline_names:
            try:
                opp = self.league._instantiate(bname).policy  # raw policy under the sampler
            except Exception:
                continue
            try:
                res = runner.run(main, opp, num_matches=int(self.tcfg.eval_matches), seed=42)
            except Exception:
                continue
            results[bname] = res
            outcome = float(res.yellow_winrate)
            # Update Elo: outcome from "main"'s perspective.
            self.league.update_elo(opponent_name=bname, outcome=outcome)
        return results

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, episode: int, metadata: dict[str, Any]) -> Path:
        """Save a full :class:`MAPPOTrainer` checkpoint at ``ckpt_dir/main_ep_<N>.pt``.

        Also prunes the checkpoint directory down to the most-recent
        ``retention_count`` ``main_ep_*.pt`` files (always preserving
        ``best.pt`` / ``best.json``) so a long training run doesn't fill
        the disk. Failures to delete are logged via telemetry but never
        propagated -- the just-written checkpoint is still useful.
        """
        meta = dict(metadata or {})
        meta.update(
            {
                "episode": int(episode),
                "update_step": int(self.update_step),
                "env_steps": int(self.env_steps),
                "stage": self.curriculum.current_stage_name,
                "run_name": self.tcfg.run_name,
            }
        )
        path = self.tcfg.checkpoint_dir / f"main_ep_{int(episode)}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = Path(self.mappo.save(path, metadata=meta))
        # Track the most recent on-disk checkpoint so the best-promotion
        # branch can copy it as a single atomic op.
        self._latest_per_episode_ckpt: Path = saved
        with contextlib.suppress(Exception):
            self._prune_old_checkpoints()
        return saved

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        """Restore the :class:`MAPPOTrainer` state. Returns the saved metadata."""
        meta = self.mappo.load(path)
        if "episode" in meta:
            self.episode_count = int(meta["episode"])
        if "update_step" in meta:
            self.update_step = int(meta["update_step"])
        if "env_steps" in meta:
            self.env_steps = int(meta["env_steps"])
        return dict(meta)

    # ------------------------------------------------------------------
    # Best-checkpoint promotion + retention
    # ------------------------------------------------------------------

    def _best_root(self) -> Path:
        """Repo-wide ``models/checkpoints`` root for shared ``best.pt``.

        We promote into the repo-shared root (parent of the run-specific
        ``checkpoint_dir``) so the API's ``best`` policy spec can find one
        deterministic file regardless of which run produced it.
        """
        # checkpoint_dir = models/checkpoints/<run>. We want models/checkpoints.
        return self.tcfg.checkpoint_dir.parent

    def _load_best_score_from_disk(self) -> None:
        """Read the persisted best.json (if any) so we don't downgrade on resume."""
        info_path = self._best_root() / "best.json"
        if not info_path.is_file():
            return
        try:
            raw = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        try:
            self._best_score = float(raw.get("score", float("-inf")))
            self._best_episode = int(raw.get("episode", -1))
        except (TypeError, ValueError):
            # Corrupted file; ignore and start fresh.
            self._best_score = float("-inf")
            self._best_episode = -1

    @staticmethod
    def _compute_combined_score(eval_results: dict[str, EvalResult]) -> float:
        """Mean winrate across all evaluated baselines (default scoring).

        Treating every baseline equally is a deliberately simple V1 choice
        so we don't have to hand-tune weights. Returns ``-inf`` when there
        are no usable results so ``_maybe_promote_best`` correctly skips
        promotion.
        """
        winrates: list[float] = []
        for res in eval_results.values():
            with contextlib.suppress(AttributeError, TypeError, ValueError):
                winrates.append(float(res.yellow_winrate))
        if not winrates:
            return float("-inf")
        return float(sum(winrates) / len(winrates))

    def _maybe_promote_best(self, eval_results: dict[str, EvalResult]) -> None:
        """Copy the latest checkpoint to ``best.pt`` when its score improves.

        Source preference: the most recently written ``main_ep_*.pt`` from
        this run. When ``save_checkpoint`` hasn't fired yet (eval interval
        is shorter than checkpoint interval) we fall back to writing a
        fresh ``best.pt`` directly via the mappo trainer so the file
        always reflects current weights.
        """
        score = self._compute_combined_score(eval_results)
        if score == float("-inf"):
            return
        if score <= self._best_score:
            return

        best_root = self._best_root()
        best_root.mkdir(parents=True, exist_ok=True)
        best_pt = best_root / "best.pt"
        best_sidecar = best_root / "best.json"

        src = self._latest_per_episode_ckpt
        if src is not None and src.is_file():
            try:
                shutil.copy2(src, best_pt)
            except OSError:
                # Fall back to a fresh save if copy fails (rare on Windows
                # when the file is being read by another process).
                src = None

        if src is None or not best_pt.is_file():
            # No fresh source ckpt -- write current weights directly.
            try:
                self.mappo.save(
                    best_pt,
                    metadata={
                        "kind": "best",
                        "episode": int(self.episode_count),
                        "update_step": int(self.update_step),
                        "env_steps": int(self.env_steps),
                        "score": float(score),
                        "scoring": "mean_winrate_vs_baselines",
                        "run_name": self.tcfg.run_name,
                        "timestamp": float(time.time()),
                    },
                )
            except Exception:
                return

        # Always (re)write the human-readable sidecar so the API can
        # surface metadata even when best.pt was produced via shutil.copy2.
        sidecar_payload: dict[str, Any] = {
            "kind": "best",
            "episode": int(self.episode_count),
            "update_step": int(self.update_step),
            "env_steps": int(self.env_steps),
            "score": float(score),
            "scoring": "mean_winrate_vs_baselines",
            "run_name": self.tcfg.run_name,
            "source_checkpoint": str(src) if src is not None else None,
            "per_baseline": {
                str(name): float(getattr(res, "yellow_winrate", float("nan")))
                for name, res in eval_results.items()
            },
            "timestamp": float(time.time()),
        }
        with contextlib.suppress(OSError):
            best_sidecar.write_text(
                json.dumps(sidecar_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        self._best_score = float(score)
        self._best_episode = int(self.episode_count)

        with contextlib.suppress(Exception):
            self.telemetry.log_text(
                "best_checkpoint",
                f"score={score:.4f} ep={self.episode_count} -> {best_pt}",
                step=self.update_step,
            )

    def _prune_old_checkpoints(self) -> None:
        """Keep only the ``retention_count`` newest ``main_ep_*.pt`` files.

        Always preserves the shared ``best.pt`` / ``best.json`` (which live
        in the parent ``models/checkpoints`` directory anyway, but be
        defensive). Each deleted .pt is paired with its ``.pt.json`` and
        ``.json`` sidecars so the directory stays tidy.
        """
        keep = int(self.tcfg.retention_count)
        if keep <= 0:
            return
        ckpt_dir = self.tcfg.checkpoint_dir
        if not ckpt_dir.is_dir():
            return
        files = [p for p in ckpt_dir.glob("main_ep_*.pt") if p.is_file() and p.name not in ("best.pt",)]
        if len(files) <= keep:
            return
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        to_delete = files[keep:]
        for victim in to_delete:
            with contextlib.suppress(OSError):
                victim.unlink()
            # mappo.save writes `<stem>.pt.json` (not `<stem>.json`) so try
            # both sidecar naming conventions for safety.
            for sidecar in (
                victim.with_suffix(victim.suffix + ".json"),
                victim.with_suffix(".json"),
            ):
                if sidecar.is_file():
                    with contextlib.suppress(OSError):
                        sidecar.unlink()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _snapshot_main(self) -> Path | None:
        """Save a :class:`PolicyBundle` snapshot of the current main policy."""
        try:
            bundle = PolicyBundle.from_kivski_config(
                self.model,
                self.active_cfg,
                metadata={
                    "kind": "snapshot",
                    "episode": int(self.episode_count),
                    "update_step": int(self.update_step),
                    "stage": self.curriculum.current_stage_name,
                },
            )
            path = self.league.snapshot_dir / f"snapshot_ep_{int(self.episode_count)}.pt"
            bundle.save(path)
            return Path(path)
        except Exception:
            return None

    def _should_trigger(self, every: int, last_at: int) -> bool:
        """Return True if we just crossed an ``every``-episode boundary."""
        e = int(every)
        if e <= 0:
            return False
        return self.episode_count > 0 and self.episode_count >= last_at + e

    def _process_episode_results(
        self,
        episode_stats: list[EpisodeStats],
        opponent_name: str,
    ) -> None:
        """Update league Elo from per-episode outcomes + log per-episode metrics."""
        for stats in episode_stats:
            # The training side is YELLOW in our setup.
            outcome = 1.0 if stats.winner == "yellow" else 0.0 if stats.winner == "blue" else 0.5
            with contextlib.suppress(Exception):
                self.league.update_elo(opponent_name=opponent_name, outcome=outcome)
            with contextlib.suppress(Exception):
                self.telemetry.log_dict(episode_stats_to_dict(stats), step=self.update_step)

    def _log_step(
        self,
        loss: MAPPOLoss,
        comm_usage: CommUsageStats,
        fps: float,
    ) -> None:
        """Push per-update training metrics to the telemetry sink."""
        # Advantage stats from the buffer for diagnostics.
        adv = self.buffer.advantages[: self.buffer.step]
        try:
            adv_mean = float(adv.mean().item()) if adv.numel() > 0 else 0.0
            adv_std = float(adv.std().item()) if adv.numel() > 1 else 0.0
        except Exception:
            adv_mean = 0.0
            adv_std = 0.0
        metrics = TrainStepMetrics(
            step=int(self.update_step),
            episode=int(self.episode_count),
            policy_loss=float(loss.policy_loss),
            value_loss=float(loss.value_loss),
            entropy=float(loss.entropy),
            kl_divergence=float(loss.kl),
            explained_variance=float(loss.explained_variance),
            grad_norm=float(loss.grad_norm),
            learning_rate=float(self.active_cfg.ml.learning_rate),
            advantage_mean=float(adv_mean),
            advantage_std=float(adv_std),
            fps=float(fps),
        )
        try:
            self.telemetry.log_dict(train_metrics_to_dict(metrics), step=self.update_step)
            self.telemetry.log_dict(comm_usage_to_dict(comm_usage), step=self.update_step)
            self.telemetry.log_dict(
                {
                    "league/main_elo": float(
                        self.league.elo_tracker.ratings.get(
                            "main",
                            type("S", (), {"rating": 1000.0})(),
                        ).rating
                    ),
                    "league/roster_size": float(len(self.league.roster)),
                    "league/episode": float(self.episode_count),
                },
                step=self.update_step,
            )
            # Consolidated "live" snapshot for the API broadcaster — keys
            # mirror the frontend MetricsSample / TrainingStatus contract
            # so the live viewer can render without any extra mapping.
            self.telemetry.log_dict(
                {
                    "live/episode": float(self.episode_count),
                    "live/total_episodes": float(self.tcfg.total_episodes),
                    "live/policy_loss": float(loss.policy_loss),
                    "live/value_loss": float(loss.value_loss),
                    "live/entropy": float(loss.entropy),
                    "live/fps": float(fps),
                    "live/kl": float(loss.kl),
                },
                step=self.update_step,
            )
        except Exception:
            pass

    def _print_progress(
        self,
        loss: MAPPOLoss,
        opponent_name: str,
        fps: float,
    ) -> None:
        """Print a short one-line summary to stdout."""
        msg = (
            f"[ep={self.episode_count}/{self.tcfg.total_episodes}] "
            f"update={self.update_step} "
            f"stage={self.curriculum.current_stage_name} "
            f"opp={opponent_name} "
            f"policy_loss={loss.policy_loss:+.3f} "
            f"value_loss={loss.value_loss:+.3f} "
            f"entropy={loss.entropy:.3f} "
            f"fps={fps:.1f}"
        )
        print(msg, flush=True)

    def _log_eval_results(self, results: dict[str, EvalResult]) -> None:
        """Forward eval results to telemetry sinks."""
        for name, res in results.items():
            with contextlib.suppress(Exception):
                self.telemetry.log_dict(
                    {
                        f"eval/{name}/yellow_winrate": float(res.yellow_winrate),
                        f"eval/{name}/avg_rounds_per_match": float(res.avg_rounds_per_match),
                        f"eval/{name}/bomb_plant_rate": float(res.bomb_plant_rate),
                        f"eval/{name}/bomb_defuse_rate": float(res.bomb_defuse_rate),
                    },
                    step=self.update_step,
                )
        # Consolidated "live" winrate record for the API broadcaster — the
        # MetricsSample frame on the frontend expects per-baseline winrates
        # on a single record so it can plot them on one axis.
        live: dict[str, float] = {"live/episode": float(self.episode_count)}
        random_res = results.get("random")
        if random_res is not None:
            live["live/winrate_vs_random"] = float(random_res.yellow_winrate)
        # Pick whichever scripted baseline was actually evaluated; the
        # default eval flow runs scripted_rush + scripted_hold.
        scripted = results.get("scripted_rush") or results.get("scripted_hold")
        if scripted is not None:
            live["live/winrate_vs_scripted"] = float(scripted.yellow_winrate)
        if len(live) > 1:
            with contextlib.suppress(Exception):
                self.telemetry.log_dict(live, step=self.update_step)

    def _handle_stage_flip(self) -> None:
        """Rebuild env / model when curriculum advances to a new stage."""
        new_cfg = self._cfg_with_safety_overrides(self.curriculum.current_config())
        # If team_size is unchanged we can keep the model + buffer (cheap path).
        old_team_size = int(self.active_cfg.simulation.team_size)
        new_team_size = int(new_cfg.simulation.team_size)
        self.active_cfg = new_cfg

        if old_team_size == new_team_size:
            # Just re-create the vec env so the new sim settings take effect.
            self.vec_env.close()
            self.vec_env = make_vec_env(
                num_envs=int(self.tcfg.num_envs),
                cfg=self.active_cfg,
                map_name=self.tcfg.map_name,
                base_seed=int(self.active_cfg.seed) + 1_000_000 + self.update_step,
                kind=str(self.tcfg.vec_kind),
                num_workers=self.tcfg.num_workers,
            )
            # The new envs were built without our curriculum gate; re-apply.
            self._apply_reward_curriculum()
            return

        # Team size changed -> rebuild everything that depends on it.
        self.vec_env.close()
        self.vec_env = make_vec_env(
            num_envs=int(self.tcfg.num_envs),
            cfg=self.active_cfg,
            map_name=self.tcfg.map_name,
            base_seed=int(self.active_cfg.seed) + 1_000_000 + self.update_step,
            kind=str(self.tcfg.vec_kind),
            num_workers=self.tcfg.num_workers,
        )
        team_size = new_team_size
        obs_dim = int(self.vec_env.obs_dim)
        action_dims = default_action_dims(team_size)
        joint_obs_dim = infer_joint_obs_dim(obs_dim, team_size)
        self.model = build_model(
            cfg=self.active_cfg,
            obs_dim=obs_dim,
            joint_obs_dim=joint_obs_dim,
            action_dims=action_dims,
            device=self.device,
        )
        self.mappo = build_trainer(self.model, self.active_cfg, device=self.device)
        self.training_runner = PolicyRunner(model=self.model, device=self.device, deterministic=False)
        self.buffer = RolloutBuffer(
            T=int(self.tcfg.rollout_steps),
            N_envs=int(self.tcfg.num_envs),
            n_agents=team_size,
            obs_dim=obs_dim,
            joint_obs_dim=joint_obs_dim,
            n_heads=int(self.vec_env.n_heads),
            hidden_size=int(self.model.hidden_size),
            comm_value_dim=int(self.model.comm_value_dim),
            device=self.device,
            n_teammates=team_size,
            gru_layers=int(self.model.gru_layers),
        )
        # Re-anchor the league on the new env / model.
        self.league.env = self.vec_env.envs[0]
        self.league.set_main_model(self.model)
        # Same logic as the cheap path -- new envs need the curriculum gate.
        self._apply_reward_curriculum()

    def _apply_reward_curriculum(self) -> None:
        """Push the current reward-curriculum stage out to every env."""
        if not self.reward_curriculum.enabled:
            return
        state = self.reward_curriculum.state
        # Never fatal: a single failed propagate must not kill the trainer loop.
        with contextlib.suppress(Exception):
            self.vec_env.set_curriculum_stage(state.stage_name, list(state.features))
        # Also log so the user can see the flip in the metrics jsonl.
        with contextlib.suppress(Exception):
            self.telemetry.log_dict(
                {
                    "live/reward_curriculum_stage": float(state.stage_index),
                    "live/reward_curriculum_stage_name": str(state.stage_name),
                },
                step=self.update_step,
            )

    @staticmethod
    def _cfg_with_safety_overrides(cfg: KivskiConfig) -> KivskiConfig:
        """Disable side-switching during training and clamp anything dangerous."""
        raw = cfg.model_dump()
        raw.setdefault("simulation", {})["side_switch_round"] = max(int(cfg.simulation.max_rounds) + 1, 999)
        return KivskiConfig.model_validate(raw)

    def _resolve_eval_scenario(self, name: str) -> ScenarioSpec:
        for spec in ALL_SCENARIOS:
            if spec.name == name:
                return spec
        # Fall back to the first scenario that has the same team size as the
        # active config so eval still runs in some sane configuration.
        for spec in ALL_SCENARIOS:
            if int(spec.team_size) == int(self.active_cfg.simulation.team_size):
                return spec
        return ALL_SCENARIOS[0]

    def _hyperparam_dump(self) -> dict[str, Any]:
        """Flatten the run's headline hyperparams for the telemetry hparams call."""
        ml = self.active_cfg.ml
        sim = self.active_cfg.simulation
        league = self.active_cfg.league
        return {
            "run_name": str(self.tcfg.run_name or "unnamed"),
            "device": str(self.device),
            "num_envs": int(self.tcfg.num_envs),
            "rollout_steps": int(self.tcfg.rollout_steps),
            "total_episodes": int(self.tcfg.total_episodes),
            "vec_kind": str(self.tcfg.vec_kind),
            "num_workers": int(self.tcfg.num_workers) if self.tcfg.num_workers else 0,
            "team_size": int(sim.team_size),
            "max_rounds": int(sim.max_rounds),
            "tick_rate_hz": int(sim.tick_rate_hz),
            "learning_rate": float(ml.learning_rate),
            "ppo_clip": float(ml.ppo_clip),
            "ppo_epochs": int(ml.ppo_epochs),
            "minibatch_size": int(ml.minibatch_size),
            "entropy_coef": float(ml.entropy_coef),
            "value_coef": float(ml.value_coef),
            "gae_lambda": float(ml.gae_lambda),
            "gamma": float(ml.gamma),
            "max_grad_norm": float(ml.max_grad_norm),
            "hidden_size": int(ml.hidden_size),
            "gru_layers": int(ml.gru_layers),
            "comm_attention_heads": int(ml.comm_attention_heads),
            "comm_embedding_dim": int(ml.comm_embedding_dim),
            "league_population_size": int(league.population_size),
            "league_snapshot_every": int(league.snapshot_every_episodes),
            "league_random_fraction": float(league.random_fraction),
            "league_scripted_fraction": float(league.scripted_fraction),
            "league_exploit_fraction": float(league.exploit_fraction),
        }
