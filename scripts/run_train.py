import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fl_privacy.data import dirichlet_partitions, get_mnist, make_client_loaders, make_test_loader
from fl_privacy.defenses import clip_by_global_norm, dp_light, laugd
from fl_privacy.fl import aggregate_mean, apply_grads, compute_grads, evaluate
from fl_privacy.models import SimpleCNN
from fl_privacy.utils import ensure_dir, get_device, set_seed, write_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default="")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--clients-per-round", type=int, default=5)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5)

    p.add_argument("--defense", type=str, default="none", choices=["none", "clipping", "dp_light", "laugd"])

    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--noise-multiplier", type=float, default=0.1)

    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--p-max", type=float, default=0.5)
    p.add_argument("--normalize-score", action="store_true")
    p.add_argument("--unbiased", action="store_true")
    p.add_argument("--mask-mode", type=str, default="bernoulli", choices=["bernoulli", "fixed_budget"])
    p.add_argument("--fixed-p", type=float, default=-1.0)

    p.add_argument("--capture-attack", action="store_true")
    p.add_argument("--attack-round", type=int, default=0)
    p.add_argument("--attack-client", type=int, default=0)
    p.add_argument("--attack-spec", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    train_ds, test_ds = get_mnist(args.data_dir)
    labels = np.array(train_ds.targets)
    parts = dirichlet_partitions(labels, num_clients=args.num_clients, alpha=args.dirichlet_alpha, seed=args.seed)
    client_loaders = make_client_loaders(
        train_dataset=train_ds,
        partitions=parts,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = make_test_loader(test_ds, batch_size=256, num_workers=args.num_workers)
    iters = {cid: iter(loader) for cid, loader in client_loaders.items()}

    model = SimpleCNN().to(device)

    run_name = args.run_name.strip() or time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir) / run_name
    ensure_dir(run_dir)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    metrics_path = run_dir / "metrics.jsonl"
    attack_obs_path = run_dir / "attack_obs.pt"

    attack_specs = []
    if args.attack_spec.strip():
        with open(args.attack_spec, "r", encoding="utf-8") as f:
            attack_specs = json.load(f)
        if not isinstance(attack_specs, list):
            raise ValueError("attack_spec must be a json list")
        for s in attack_specs:
            if "round" not in s or "client_id" not in s:
                raise ValueError("each attack spec must contain round and client_id")
            if "tag" not in s:
                s["tag"] = f"r{s['round']}_c{s['client_id']}"

    for r in range(args.rounds):
        rng = np.random.default_rng(args.seed + r)
        chosen = rng.choice(args.num_clients, size=min(args.clients_per_round, args.num_clients), replace=False).tolist()
        must_include = []
        if args.capture_attack:
            if attack_specs:
                for s in attack_specs:
                    if int(s["round"]) == r:
                        must_include.append(int(s["client_id"]))
            else:
                if r == args.attack_round:
                    must_include.append(int(args.attack_client))

        if must_include:
            must_include = list(dict.fromkeys(must_include))
            for i, cid in enumerate(must_include):
                if cid in chosen:
                    continue
                if i < len(chosen):
                    chosen[i] = cid
                else:
                    chosen.append(cid)

        updates = []
        round_losses = []
        round_accs = []

        for cid in chosen:
            try:
                batch = next(iters[cid])
            except StopIteration:
                iters[cid] = iter(client_loaders[cid])
                batch = next(iters[cid])

            loss, logits, grads, x, y = compute_grads(model, batch=batch, device=device)
            upd = grads

            defense_stats = None
            if args.defense == "clipping":
                upd, defense_stats = clip_by_global_norm(upd, max_norm=args.clip_norm)
            elif args.defense == "dp_light":
                upd, defense_stats = dp_light(upd, clip_norm=args.clip_norm, noise_multiplier=args.noise_multiplier)
            elif args.defense == "laugd":
                fixed_p = None if args.fixed_p < 0 else float(args.fixed_p)
                upd, defense_stats = laugd(
                    upd,
                    alpha=args.alpha,
                    tau=args.tau,
                    p_max=args.p_max,
                    normalize_score=bool(args.normalize_score),
                    unbiased=bool(args.unbiased),
                    mask_mode=args.mask_mode,
                    fixed_p=fixed_p,
                )

            if args.capture_attack and r == args.attack_round and cid == args.attack_client:
                payload = {
                    "round": r,
                    "client_id": cid,
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "x": x.cpu(),
                    "y": y.cpu(),
                    "grads_obs": [t.detach().cpu() for t in upd],
                    "defense": args.defense,
                    "defense_stats": None if defense_stats is None else defense_stats.__dict__,
                }
                torch.save(payload, attack_obs_path)
            if args.capture_attack and attack_specs:
                for s in attack_specs:
                    if int(s["round"]) == r and int(s["client_id"]) == cid:
                        payload = {
                            "round": r,
                            "client_id": cid,
                            "tag": str(s["tag"]),
                            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "x": x.cpu(),
                            "y": y.cpu(),
                            "grads_obs": [t.detach().cpu() for t in upd],
                            "defense": args.defense,
                            "defense_stats": None if defense_stats is None else defense_stats.__dict__,
                        }
                        torch.save(payload, run_dir / f"attack_obs_{s['tag']}.pt")

            updates.append(upd)
            round_losses.append(float(loss))
            round_accs.append(float((logits.argmax(dim=1) == y).float().mean().item()))

        agg = aggregate_mean(updates)
        apply_grads(model, agg, lr=args.lr)

        if (r + 1) % args.eval_every == 0 or r == 0:
            test_loss, test_acc = evaluate(model, test_loader, device=device)
        else:
            test_loss, test_acc = float("nan"), float("nan")

        row = {
            "round": r,
            "train_loss": float(np.mean(round_losses)) if round_losses else float("nan"),
            "train_acc": float(np.mean(round_accs)) if round_accs else float("nan"),
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        write_jsonl(metrics_path, row)

    torch.save(model.state_dict(), run_dir / "model.pt")
    print(str(run_dir))


if __name__ == "__main__":
    main()
