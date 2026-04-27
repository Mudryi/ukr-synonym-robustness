"""Unified CLI for running TextFooler or BERT-Attack on a dataset.

Example:

    python -m src.cli.run_attack \\
        --attack textfooler --dataset reviews \\
        --dataset-path data/reviews/test_reviews.csv \\
        --target-model xlm-roberta-base \\
        --target-checkpoint /path/to/finetuned/ckpt \\
        --nclasses 5 \\
        --synonym-dict /path/to/synonimy_info_clean.json \\
        --hand-parsed /path/to/hand_parsed_top_100.json \\
        --antonyms /path/to/antonimy.jsonlines \\
        --output-dir results/textfooler_reviews_xlmr/

A YAML ``--config`` may pre-fill dataset-level defaults; CLI flags override.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm


# ---------- helpers ----------

def _load_yaml(path: str | Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_textfooler(args):
    from ..core.predictor import Predictor
    from ..core.similarity import SBERT
    from ..core.stopwords import get_stopwords
    from ..core.synonym_dict import read_and_clean_synonym_dict
    from ..attacks.textfooler import TextFoolerAttack

    syn_path = args.synonym_dict or os.getenv("UKR_SYN_DICT")
    if not syn_path:
        sys.exit(
            "TextFooler requires --synonym-dict (or UKR_SYN_DICT env var). "
            "See data/synonyms/README.md for the expected format."
        )
    hand = args.hand_parsed or os.getenv("UKR_SYN_HAND")
    ant = args.antonyms or os.getenv("UKR_SYN_ANTONYMS")

    print(f"loading synonym dict from {syn_path}")
    synonym_dict = read_and_clean_synonym_dict(
        syn_path, hand_parsed_path=hand, antonyms_path=ant
    )

    predictor = Predictor(
        target_model=args.target_model,
        target_checkpoint=args.target_checkpoint,
        nclasses=args.nclasses,
    )
    sbert = SBERT(args.sbert_model)
    return TextFoolerAttack(
        predictor=predictor,
        sbert=sbert,
        synonym_dict=synonym_dict,
        stopwords=get_stopwords(),
        sim_threshold=args.sim_threshold,
        synonym_num=args.synonym_num,
    ), predictor


def _build_bert_attack(args):
    from transformers import AutoTokenizer, RobertaConfig, RobertaForMaskedLM, pipeline

    from ..core.predictor import Predictor
    from ..core.similarity import FastTextSim, SBERT
    from ..core.stopwords import get_stopwords
    from ..attacks.bert_attack import BertAttack

    predictor = Predictor(
        target_model=args.target_model,
        target_checkpoint=args.target_checkpoint,
        nclasses=args.nclasses,
    )

    tokenizer_mlm = AutoTokenizer.from_pretrained(args.mlm_model, truncation=True)
    tokenizer_mlm.model_max_length = 512
    config_atk = RobertaConfig.from_pretrained(args.mlm_model)
    mlm_model = RobertaForMaskedLM.from_pretrained(args.mlm_model, config=config_atk).to(
        predictor.device
    )

    unmasker = None
    if args.bert_mode == "fill-mask":
        unmasker = pipeline("fill-mask", model=args.mlm_model, tokenizer=tokenizer_mlm)

    ft_sim = FastTextSim(args.fasttext_path)

    sbert = None
    if args.sbert_model:
        sbert = SBERT(args.sbert_model)

    attack = BertAttack(
        predictor=predictor,
        mlm_model=mlm_model,
        tokenizer_mlm=tokenizer_mlm,
        stopwords=get_stopwords(),
        ft_sim=ft_sim,
        unmasker=unmasker,
        mode=args.bert_mode,
        sbert=sbert,
        cos_sim_threshold=args.cos_sim_threshold,
        max_changes_frac=args.max_changes_frac,
        topk=args.bert_topk,
        threshold_pred_score=args.bert_threshold_pred_score,
        num_subs=args.bert_num_subs,
        threshold_score=args.bert_threshold_score,
    )
    return attack, predictor


# ---------- argparse ----------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run a synonym-substitution attack.")
    p.add_argument("--config", type=str, help="Optional YAML config providing defaults.")

    p.add_argument("--attack", choices=["textfooler", "bert_attack"], required=False)
    p.add_argument("--dataset", choices=["reviews", "news", "unlp"], required=False)
    p.add_argument("--dataset-path", type=str, required=False)
    p.add_argument("--target-model", type=str, required=False,
                   help="HuggingFace name for the tokenizer.")
    p.add_argument("--target-checkpoint", type=str, required=False,
                   help="Path to the finetuned classifier checkpoint.")
    p.add_argument("--nclasses", type=int, required=False)
    p.add_argument("--output-dir", type=str, required=False)

    p.add_argument("--n-samples", type=int, default=None,
                   help="If set, only attack this many examples.")
    p.add_argument("--seed", type=int, default=1914)

    # TextFooler
    p.add_argument("--sbert-model", type=str,
                   default="sentence-transformers/paraphrase-xlm-r-multilingual-v1")
    p.add_argument("--synonym-dict", type=str)
    p.add_argument("--hand-parsed", type=str)
    p.add_argument("--antonyms", type=str)
    p.add_argument("--sim-threshold", type=float, default=0.7)
    p.add_argument("--synonym-num", type=int, default=200)

    # BERT-Attack
    p.add_argument("--bert-mode", choices=["classic", "fill-mask"], default="fill-mask")
    p.add_argument("--mlm-model", type=str, default="FacebookAI/xlm-roberta-large")
    p.add_argument("--fasttext-path", type=str)
    p.add_argument("--cos-sim-threshold", type=float, default=0.33)
    p.add_argument("--max-changes-frac", type=float, default=0.4)
    p.add_argument("--bert-topk", type=int, default=48)
    p.add_argument("--bert-threshold-pred-score", type=float, default=60.0)
    p.add_argument("--bert-num-subs", type=int, default=128)
    p.add_argument("--bert-threshold-score", type=float, default=0.04)

    return p.parse_args(argv)


def _merge_config(args, cfg: dict) -> argparse.Namespace:
    """Apply YAML defaults for any flag the user did not pass on the CLI."""
    for k, v in cfg.items():
        # convert hyphenated YAML key style transparently
        attr = k.replace("-", "_")
        if not hasattr(args, attr):
            continue
        if getattr(args, attr) in (None, False):
            setattr(args, attr, v)
    return args


# ---------- main ----------

def main(argv=None):
    import random

    import numpy as np
    import torch

    args = parse_args(argv)

    if args.config:
        cfg = _load_yaml(args.config)
        args = _merge_config(args, cfg)

    missing = [
        flag for flag, val in [
            ("--attack", args.attack),
            ("--dataset", args.dataset),
            ("--dataset-path", args.dataset_path),
            ("--target-model", args.target_model),
            ("--target-checkpoint", args.target_checkpoint),
            ("--nclasses", args.nclasses),
            ("--output-dir", args.output_dir),
        ] if val in (None, "")
    ]
    if missing:
        sys.exit(f"missing required flags: {', '.join(missing)}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from ..core.data import load_dataset
    from ..evaluation.writer import ResultWriter

    print(f"loading dataset {args.dataset} from {args.dataset_path}")
    data = load_dataset(args.dataset, args.dataset_path, sample_seed=args.seed)
    if args.n_samples is not None:
        data = data[: args.n_samples]
    print(f"will attack {len(data)} examples")

    print(f"building attack: {args.attack}")
    if args.attack == "textfooler":
        attack, _ = _build_textfooler(args)
    else:
        attack, _ = _build_bert_attack(args)

    output_dir = Path(args.output_dir)
    config_dump = vars(args).copy()
    config_dump.pop("config", None)

    started = time.time()
    with ResultWriter(output_dir) as writer:
        for idx, (text, label) in enumerate(tqdm(data, desc=attack.name)):
            try:
                result = attack.run(text, label, idx)
            except Exception as exc:  # noqa: BLE001 — log and continue on per-sample failure
                print(f"[{idx}] attack failed with {exc!r}; recording as FAILED")
                from ..attacks.base import AttackResult, Status
                result = AttackResult(
                    id=idx, attack=attack.name, orig_text=text, adv_text=text,
                    true_label=int(label), orig_label=int(label), adv_label=int(label),
                    status=Status.FAILED.value, num_changes=0, num_queries=0,
                    change_rate=0.0, replacements=[], semantic_sim=None,
                )
            writer.write(result)

        wall = time.time() - started
        summary = writer.finalize(
            attack=attack.name,
            dataset=args.dataset,
            target_model=args.target_model,
            target_checkpoint=args.target_checkpoint,
            config=config_dump,
            wall_time_sec=wall,
        )

    print(f"\nfinished in {wall:.1f}s")
    print(
        f"original_acc={summary.original_accuracy:.3f} "
        f"after_atk_acc={summary.after_attack_accuracy:.3f} "
        f"ASR={summary.attack_success_rate:.3f} "
        f"avg_queries={summary.avg_queries:.1f} "
        f"avg_change_rate={summary.avg_change_rate:.3f}"
    )
    print(f"summary: {writer.summary_path}")
    print(f"examples: {writer.examples_path}")
    print(f"appended to: {writer.runs_csv}")


if __name__ == "__main__":
    main()
