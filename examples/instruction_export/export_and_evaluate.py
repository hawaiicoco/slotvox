#!/usr/bin/env python3
"""Export instruction data and evaluate replayed decisions (torch-free).

Pipeline: synthetic dataset -> instruction corpus (JSONL + manifest) ->
gold-replay fixtures -> adapter-served decisions -> scored run envelope
-> Markdown/HTML reports.

Honest scope: the replay fixtures are built FROM the gold annotations,
so the evaluation scores come out perfect BY CONSTRUCTION — this
example validates the serving + evaluation plumbing end to end, not
model quality.

Usage:
    python examples/instruction_export/export_and_evaluate.py --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slotvox.adapters.protocol import InferRequest, InferResponse
from slotvox.adapters.replay import Fixture, ReplayAdapter, save_fixtures
from slotvox.config import FeatureConfig, GenerationConfig
from slotvox.data.alignment import frame_span_tags, frame_token_map
from slotvox.data.dataset import generate_dataset
from slotvox.eval.metrics import TurnRecord, slice_pairs
from slotvox.eval.report import html_from_envelope, markdown_from_envelope, write_report_files
from slotvox.eval.runs import ScoredRun
from slotvox.instructions.export import (
    export_instructions,
    load_instructions,
    read_export_manifest,
)
from slotvox.instructions.templates import render_sample
from slotvox.schema.serialize import write_json_atomic


def main() -> int:
    """Run the full data pipeline and write reports under --out."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--domain", default="weather")
    parser.add_argument("--language", default="zh", choices=("zh", "en"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    config = GenerationConfig(
        domain=args.domain, counts={"train": 4, "dev": 2}, n_speakers=2, seed=args.seed
    )
    dataset = generate_dataset(config, language=args.language)
    print(f"dataset: {len(dataset.examples)} synthetic utterances, hash {dataset.dataset_hash}")

    samples = [
        render_sample(example, domain=dataset.domain_id, dataset_hash=dataset.dataset_hash)
        for example in dataset.examples
    ]
    corpus_dir = export_instructions(samples, args.out / "corpus", overwrite=True)
    manifest = read_export_manifest(corpus_dir)
    assert len(load_instructions(corpus_dir)) == manifest["sample_count"]
    print(
        f"corpus: {manifest['sample_count']} samples, "
        f"quality passed {manifest['quality']['passed']}/{manifest['sample_count']}"
    )

    feature_config = FeatureConfig()
    fixtures = []
    for example in dataset.examples:
        token_map = frame_token_map(example.utterance, feature_config)
        gold_tags = frame_span_tags(example.annotation.tags, token_map)
        request = InferRequest(
            example.utterance_id,
            tuple(float(value) for value in example.utterance.samples),
            feature_config.sample_rate,
        )
        response = InferResponse(
            request.request_id,
            example.annotation.intent,
            1.0,
            gold_tags,
            dataset.dataset_hash,  # no model here; the dataset hash pins provenance
        )
        fixtures.append(Fixture(request, response))
    adapter = ReplayAdapter(fixtures)
    save_fixtures(fixtures, args.out / "fixtures.json")
    print(f"fixtures: {len(fixtures)} gold-replay pairs (scores below are perfect BY CONSTRUCTION)")

    records = []
    for example, fixture in zip(dataset.examples, fixtures, strict=True):
        response = adapter.infer(fixture.request)
        token_map = frame_token_map(example.utterance, feature_config)
        gold_tags = frame_span_tags(example.annotation.tags, token_map)
        noise = "clean" if example.snr_db is None else f"snr-{int(example.snr_db)}"
        records.append(
            TurnRecord(
                example.annotation.intent,
                response.intent,
                gold_tags,
                tuple(response.frame_tags),
                slices=slice_pairs({"noise": noise, "speaker": f"spk-{example.speaker_id}"}),
            )
        )
    run = ScoredRun.evaluate(
        "example-replay",
        records,
        metadata={"dataset_hash": dataset.dataset_hash, "mode": "gold-replay"},
        bootstrap_seed=args.seed,
        replicates=200,
    )
    write_json_atomic(args.out / "run.json", run.to_dict())

    envelope = run.to_dict()
    report_dir = write_report_files(
        markdown_from_envelope(envelope),
        html_from_envelope(envelope),
        args.out / "report",
        overwrite=True,
    )
    print(f"intent accuracy: {run.intent.accuracy:.4f} (gold replay)")
    print(f"slot f1 strict/partial: {run.slot_strict.f1:.4f}/{run.slot_partial.f1:.4f}")
    print(f"reports: {report_dir}")
    print(
        "note: perfect scores validate the plumbing, not any model — the "
        "fixtures replay gold decisions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
