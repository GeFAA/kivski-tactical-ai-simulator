"""Train a KivskiActorCritic on imitation demos and save as PolicyBundle.

Three blocks of work:
1. Build the model from the configured arch (configs/production.yaml by
   default) so its shape matches the cloud trainer exactly.
2. Loop over the imitation dataset for N epochs with Adam, minimising the
   negative log-prob of the demo (move + 4 discrete) actions under the
   policy -- i.e. exactly what KivskiActorCritic.evaluate() returns.
3. Save the trained state as a PolicyBundle so the existing
   `kivski-train train --resume` flag accepts it.

Why use `model.evaluate(...)` instead of a custom bc_forward?
  Because evaluate() already implements the *exact* trunk-call sequence
  (obs_encoder -> GRU -> actor_heads.evaluate) used at training time. Any
  bespoke bc_forward would risk mirroring drift the moment the act/eval
  trunk evolves. evaluate() returns log_probs; BC loss = -log_probs.mean().

Usage:
    python -m scripts.train_bc \
        --demos data/imitation_demos.pt \
        --epochs 30 \
        --batch-size 256 \
        --out models/checkpoints/bc_pretrained.pt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader

from kivski_agents.factory import build_model, default_action_spec, infer_joint_obs_dim
from kivski_agents.imitation.dataset import ImitationDataset
from kivski_agents.policy_runner import PolicyBundle
from kivski_sim.config import load_config
from kivski_sim.obs_decoder import get_observation_dim

app = typer.Typer(add_completion=False)


@app.command()
def train(
    demos: str = typer.Option("data/imitation_demos.pt", "--demos"),
    config: str = typer.Option("configs/production.yaml", "--config", "-c"),
    epochs: int = typer.Option(30, "--epochs"),
    batch_size: int = typer.Option(256, "--batch-size"),
    learning_rate: float = typer.Option(3.0e-4, "--learning-rate"),
    grad_clip: float = typer.Option(0.5, "--grad-clip"),
    out: str = typer.Option("models/checkpoints/bc_pretrained.pt", "--out"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """BC-train the policy on collected demos."""
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    cfg = load_config(config)

    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    typer.echo(f"[bc] device={torch_device}")

    obs_dim = get_observation_dim(cfg)
    team_size = int(cfg.simulation.team_size)
    joint_obs_dim = infer_joint_obs_dim(obs_dim, team_size)
    action_spec = default_action_spec(team_size)

    model = build_model(cfg, obs_dim, joint_obs_dim, action_spec, device=torch_device)
    n_params = sum(p.numel() for p in model.parameters())
    typer.echo(
        f"[bc] model: hidden_size={cfg.ml.hidden_size} params={n_params:_} "
        f"obs_dim={obs_dim} joint_obs_dim={joint_obs_dim}"
    )

    demos_dict = torch.load(demos, weights_only=False)
    dataset = ImitationDataset(demos_dict)
    typer.echo(f"[bc] demos: {len(dataset):_} steps loaded from {demos}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        t0 = time.perf_counter()
        n_seen = 0
        nll_sum = 0.0
        entropy_sum = 0.0
        for batch in loader:
            obs_b = batch["obs"].to(torch_device)
            joint_b = batch["joint_obs"].to(torch_device)
            comm_b = batch["received_comm"].to(torch_device)
            move_b = batch["move"].to(torch_device)
            disc_b = batch["discrete"].to(torch_device)
            # initial_hidden_state returns [num_layers, B, hidden_size]
            B = obs_b.shape[0]
            hidden_b = model.initial_hidden_state(B, device=torch_device)

            # ---- BC forward via trunk + manual heads (teacher forcing) -------
            # We don't use model.evaluate() because its Gaussian NLL term
            # collapses (sigma -> floor) when ~half the demos are move=[0,0]
            # (buy phase). Collapse pathology: log_density blows up, NLL goes
            # to -inf, policy memorizes "always be still" and never explores.
            #
            # Instead: MSE on the move mean (the policy gradient that drives
            # log_std stays untouched, so log_std stays near its init and
            # PPO can adapt it later) + per-head CE on discrete with teacher
            # forcing on previous-action embeddings.
            actor_hidden, _ = model._forward_core(obs_b, hidden_b, comm_b)
            move_mean, _move_std = model.actor_heads._move_params(actor_hidden)
            move_loss = F.mse_loss(move_mean, move_b)

            disc_loss = torch.zeros((), device=torch_device)
            disc_ce_per_head: list[float] = []
            prev_embeds: list[torch.Tensor] = []
            for i, (head, emb) in enumerate(
                zip(model.actor_heads.heads, model.actor_heads.embeddings, strict=False)
            ):
                head_in = torch.cat([actor_hidden, move_b, *prev_embeds], dim=-1)
                logits = head(head_in)
                ce_i = F.cross_entropy(logits, disc_b[:, i])
                disc_loss = disc_loss + ce_i
                disc_ce_per_head.append(float(ce_i.detach().cpu().item()))
                prev_embeds.append(emb(disc_b[:, i]))

            loss = move_loss + disc_loss

            # For monitoring only -- log entropy via the standard evaluate
            # path every Nth step (cheap if Nth, expensive every step).
            with torch.no_grad():
                if n_seen % (B * 50) == 0:
                    _eval = model.evaluate(
                        obs=obs_b,
                        hidden_state=hidden_b,
                        received_comm=comm_b,
                        prev_actions={"move": move_b, "discrete": disc_b},
                        joint_obs=joint_b,
                    )
                    entropy = _eval["entropy"]
                else:
                    entropy = torch.zeros(B, device=torch_device)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

            n_seen += B
            nll_sum += float(loss.detach().cpu().item()) * B
            entropy_sum += float(entropy.detach().cpu().mean().item()) * B

        dt = time.perf_counter() - t0
        avg_loss = nll_sum / max(n_seen, 1)
        avg_ent = entropy_sum / max(n_seen, 1)
        # Last-batch per-head CE for visibility
        ce_str = "/".join(f"{c:.3f}" for c in disc_ce_per_head) if disc_ce_per_head else "-"
        typer.echo(
            f"[bc] epoch {epoch + 1:3d}/{epochs} loss={avg_loss:.4f} "
            f"move_mse={float(move_loss.item()):.4f} ce={ce_str} "
            f"avg_entropy={avg_ent:.3f} ({n_seen:_} steps in {dt:.1f}s)"
        )

    # ---- Save as PolicyBundle (compatible with kivski-train --resume) -----
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    bundle = PolicyBundle.from_kivski_config(
        model=model,
        cfg=cfg,
        metadata={
            "episode": 0,
            "update_step": 0,
            "env_steps": 0,
            "score": None,
            "scoring": "imitation_bc",
            "source": "behavior_cloning",
            "demo_steps": len(dataset),
            "bc_epochs": epochs,
            "bc_learning_rate": learning_rate,
            "bc_final_loss": avg_loss,
            "bc_loss_type": "mse_move + ce_discrete (teacher forced)",
        },
    )
    bundle.save(out_path)
    typer.echo(f"[bc] saved bundle -> {out_path} (+ sidecar .json)")


if __name__ == "__main__":
    app()
