"""Evaluate direct and symmetric-head source-to-source PQ-STF association."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from embedding_aware_belt_fusion.data.complete_cache import IndexedCompleteFrameDataset
from embedding_aware_belt_fusion.embeddings.all_agent_association import (
    SymmetricQueryStatePairHead,
    all_agent_pairs,
    binary_pair_loss_and_metrics,
    direct_scores,
    load_pq_stf,
    pair_head_scores,
    records_for_role,
    shared_identity_loss_and_metrics,
    source_agent_pairs,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trackformer-root", type=Path, default=Path("external/trackformer"))
    parser.add_argument("--method", choices=("direct", "symmetric-pair"), required=True)
    parser.add_argument("--pair-head-checkpoint", type=Path)
    parser.add_argument("--scenario-split-file", type=Path)
    parser.add_argument("--scenario-role", choices=("train", "validation"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--index-workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--candidate-distance", type=float, default=5.0)
    parser.add_argument("--match-probability", type=float, default=0.90)
    parser.add_argument("--hard-negative-ratio", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _collate(frames):
    return frames


def main():
    args = parse_args()
    if args.method == "symmetric-pair" and args.pair_head_checkpoint is None:
        raise ValueError("--pair-head-checkpoint is required for --method symmetric-pair")
    if bool(args.scenario_split_file) != bool(args.scenario_role):
        raise ValueError("--scenario-split-file and --scenario-role must be supplied together")
    if args.workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device(args.device)
    records = records_for_role(
        args.cache_dir, args.scenario_split_file, args.scenario_role, args.index_workers
    )
    if args.max_frames is not None:
        records = records[: args.max_frames]
    dataset = IndexedCompleteFrameDataset(args.cache_dir, records)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_collate,
        pin_memory=False,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )
    base, saved = load_pq_stf(
        args.checkpoint,
        args.trackformer_root,
        int(dataset.manifest["feature_dim"]),
        device,
    )
    head = None
    if args.method == "symmetric-pair":
        pair_saved = torch.load(args.pair_head_checkpoint, map_location="cpu", weights_only=False)
        expected = str(args.checkpoint.resolve())
        if pair_saved.get("base_checkpoint") != expected:
            raise ValueError("Pair-head checkpoint was trained from a different PQ-STF checkpoint")
        if pair_saved.get("pipeline") != "calibrated_pq_query_state_pair_head_v2":
            raise ValueError(
                "This is an older retrieval-only pair head. Train the calibrated "
                "same-object/no-match pair head before evaluating it."
            )
        head = SymmetricQueryStatePairHead(int(pair_saved["state_dim"])).to(device)
        head.load_state_dict(pair_saved["model_state_dict"])
        head.eval()
    autocast = (
        lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        if args.amp and device.type == "cuda"
        else nullcontext()
    )
    correct, total, shared, retrieval_pairs, losses = 0.0, 0, 0, 0, []
    true_positive = false_positive = false_negative = 0
    candidate_pairs = positive_pairs = binary_agent_pairs = 0
    with torch.no_grad():
        progress = tqdm(loader, desc=f"all-agent {args.method}")
        for frames in progress:
            for frame in frames:
                pairs = (
                    source_agent_pairs(frame)
                    if args.method == "direct"
                    else all_agent_pairs(frame)
                )
                for agent_a, agent_b in pairs:
                    with autocast():
                        output_a = base.forward_agent(agent_a)
                        output_b = base.forward_agent(agent_b)
                        scores = (
                            direct_scores(output_a, output_b)
                            if head is None
                            else pair_head_scores(head, output_a, output_b)
                        )
                        record = (
                            shared_identity_loss_and_metrics(
                                scores, agent_a, agent_b, args.temperature
                            )
                            if args.method == "direct"
                            else binary_pair_loss_and_metrics(
                                scores,
                                agent_a,
                                agent_b,
                                maximum_distance=args.candidate_distance,
                                probability_threshold=args.match_probability,
                                negative_ratio=args.hard_negative_ratio,
                            )
                        )
                    if args.method == "direct" and record["top1_total"]:
                        correct += record["top1_correct"]
                        total += record["top1_total"]
                        shared += record["shared_objects"]
                        retrieval_pairs += 1
                        losses.append(float(record["loss"]))
                    elif args.method == "symmetric-pair" and record["pairs"]:
                        true_positive += record["true_positive"]
                        false_positive += record["false_positive"]
                        false_negative += record["false_negative"]
                        candidate_pairs += record["pairs"]
                        positive_pairs += record["positive_pairs"]
                        binary_agent_pairs += 1
                        losses.append(float(record["loss"]))
            if args.method == "direct":
                progress.set_postfix(top1=f"{correct / max(total, 1):.3f}")
            else:
                precision = true_positive / max(true_positive + false_positive, 1)
                recall = true_positive / max(true_positive + false_negative, 1)
                progress.set_postfix(precision=f"{precision:.3f}", recall=f"{recall:.3f}")
    if args.method == "direct" and not total:
        raise RuntimeError("No source-to-source shared object pairs were found")
    if args.method == "symmetric-pair" and not candidate_pairs:
        raise RuntimeError("No geometrically valid CAV proposal pairs were found")
    result = {
        "method": args.method,
        "checkpoint": str(args.checkpoint.resolve()),
        "pair_head_checkpoint": str(args.pair_head_checkpoint.resolve()) if args.pair_head_checkpoint else None,
        "cache": str(args.cache_dir.resolve()),
        "scenario_role": args.scenario_role,
        "frames": len(dataset),
    }
    if args.method == "direct":
        result.update({
            "source_agent_pairs": retrieval_pairs,
            "mean_shared_objects_per_source_pair": shared / retrieval_pairs,
            "cross_source_top1": correct / total,
            "cross_source_loss": mean(losses),
            "evaluation": (
                "Retrieval diagnostic: for every non-ego CAV pair, retrieve the "
                "best target proposal for each labelled source proposal."
            ),
        })
    else:
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        result.update({
            "candidate_distance_m": args.candidate_distance,
            "match_probability": args.match_probability,
            "agent_pairs": binary_agent_pairs,
            "candidate_pairs": candidate_pairs,
            "positive_pairs": positive_pairs,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-8),
            "binary_pair_loss": mean(losses),
            "evaluation": (
                "Calibrated same-object/no-match diagnostic over every CAV pair. "
                "Only proposal pairs within candidate_distance_m are considered; "
                "unmatched proposals and different physical IDs are negatives."
            ),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print("ALL_AGENT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
