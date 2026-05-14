#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.project_config import settings
from scraping_options import update_eventual_sale_labels_for_csv


def simple_scrape_dir() -> Path:
    return Path(str(settings.paths.simple_scrape_dir))


def search_dir(folder: str) -> Path:
    return simple_scrape_dir() / folder


def run_main_command(args: argparse.Namespace) -> int:
    from main import main as app_main
    return app_main()


def run_update_eventual_sales(args: argparse.Namespace) -> int:
    base = search_dir(args.folder)
    input_path = Path(args.input) if args.input else base / ("pipeline_out" if args.use_pipeline_out else "") / ("deals_ranked.csv" if args.use_pipeline_out else "big_raw.csv")
    out_dir = Path(args.out_dir) if args.out_dir else base / args.eventual_out_dir
    exclude_known_sold_csv = args.exclude_known_sold_csv
    if exclude_known_sold_csv is None and input_path.name != 'sold_df.csv':
        candidate = base / 'sold_df.csv'
        if candidate.exists():
            exclude_known_sold_csv = str(candidate)
    result = update_eventual_sale_labels_for_csv(
        str(input_path),
        out_dir=str(out_dir),
        max_workers=args.max_workers,
        delay=args.delay,
        min_deal_score=args.min_deal_score,
        min_deal_confidence=args.min_deal_confidence,
        top_n=args.top_n,
        require_deal_eligible=args.require_deal_eligible,
        sort_by=args.sort_by,
        allow_residential_fallback=args.allow_residential_fallback,
        initial_delay=args.initial_delay,
        fetch_sleep=args.fetch_sleep,
        fetch_max_attempts=args.fetch_max_attempts,
        no_residential=args.no_residential,
        recheck_sold_rows=args.recheck_sold_rows,
        exclude_known_sold_csv=exclude_known_sold_csv,
    )
    print(result)
    return 0


def run_batch_command(args: argparse.Namespace) -> int:
    base = search_dir(args.folder)
    out_dir = Path(args.out_dir) if args.out_dir else base / args.pipeline_out_dir
    db_path = Path(args.db) if args.db else out_dir / 'index.sqlite'
    input_path = Path(args.input) if args.input else base / args.input_name
    ns = SimpleNamespace(
        input=str(input_path),
        out_dir=str(out_dir),
        db=str(db_path),
        model=args.model,
        product_threshold=args.product_threshold,
        use_image_embeddings=args.use_image_embeddings,
        image_embedding_model=args.image_embedding_model,
        image_embedding_timeout=args.image_embedding_timeout,
        image_embedding_batch_size=args.image_embedding_batch_size,
        product_image_weight=args.product_image_weight,
        autotune_variants=args.autotune_variants,
        variant_threshold=args.variant_threshold,
        core_frac=args.core_frac,
        variant_price_weight=args.variant_price_weight,
        variant_image_weight=args.variant_image_weight,
        min_product_size_for_variants=args.min_product_size_for_variants,
        min_variant_size_for_deals=args.min_variant_size_for_deals,
        min_variant_size_for_confident_deals=args.min_variant_size_for_confident_deals,
        min_variant_silhouette=args.min_variant_silhouette,
        max_variant_mad_ratio=args.max_variant_mad_ratio,
        hard_max_variant_mad_ratio=args.hard_max_variant_mad_ratio,
        min_deal_confidence=args.min_deal_confidence,
        min_centroid_similarity=args.min_centroid_similarity,
        min_informative_tokens=args.min_informative_tokens,
        exclude_negflag_deals=args.exclude_negflag_deals,
        resale_fee_rate=args.resale_fee_rate,
        resale_fixed_cost=args.resale_fixed_cost,
        resale_safety_discount=args.resale_safety_discount,
        min_expected_profit=args.min_expected_profit,
        min_expected_profit_margin=args.min_expected_profit_margin,
        price_buffer_size=args.price_buffer_size,
        make_plots=args.make_plots,
    )
    from experiments.clustering_approach.vinted_pipeline_batch import run as run_batch
    run_batch(ns)
    return 0


def run_incremental_command(args: argparse.Namespace) -> int:
    base = search_dir(args.folder)
    out_dir = Path(args.out_dir) if args.out_dir else base / args.incremental_out_name
    db_path = Path(args.db) if args.db else base / args.pipeline_out_dir / 'index.sqlite'
    ns = SimpleNamespace(
        db=str(db_path),
        out=str(out_dir),
        new_items=args.new_items,
        growing_csv=args.growing_csv,
        append=args.append,
        price_buffer_size=args.price_buffer_size,
        min_variant_size_for_confident_deals=args.min_variant_size_for_confident_deals,
        max_variant_mad_ratio=args.max_variant_mad_ratio,
        hard_max_variant_mad_ratio=args.hard_max_variant_mad_ratio,
        min_deal_confidence=args.min_deal_confidence,
        min_centroid_similarity=args.min_centroid_similarity,
        min_informative_tokens=args.min_informative_tokens,
        exclude_negflag_deals=args.exclude_negflag_deals,
        resale_fee_rate=args.resale_fee_rate,
        resale_fixed_cost=args.resale_fixed_cost,
        resale_safety_discount=args.resale_safety_discount,
        min_expected_profit=args.min_expected_profit,
        min_expected_profit_margin=args.min_expected_profit_margin,
    )
    if bool(ns.new_items) == bool(ns.growing_csv):
        raise SystemExit('Provide exactly one of --new_items or --growing_csv for incremental')
    from experiments.clustering_approach.vinted_pipeline_incremental import run as run_incremental
    run_incremental(ns)
    return 0


def run_evaluate_command(args: argparse.Namespace) -> int:
    base = search_dir(args.folder)
    deals = Path(args.deals) if args.deals else base / args.pipeline_out_dir / 'deals_ranked.csv'
    sold = Path(args.sold) if args.sold else base / 'sold_df.csv'
    sold_eventually = None
    if args.sold_eventually:
        sold_eventually = Path(args.sold_eventually)
    elif args.use_eventual_sales:
        sold_eventually = base / args.eventual_out_dir / 'sold_eventually.csv'
    out_dir = Path(args.out_dir) if args.out_dir else base / args.eval_out_dir
    argv = [
        'evaluate_deal_score.py',
        '--deals', str(deals),
        '--sold', str(sold),
        '--out_dir', str(out_dir),
        '--high_score_threshold', str(args.high_score_threshold),
        '--low_score_threshold', str(args.low_score_threshold),
    ]
    if sold_eventually is not None:
        argv.extend(['--sold_eventually', str(sold_eventually)])
    if args.no_dedupe:
        argv.append('--no_dedupe')
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        from analysis_pipeline.evaluation.evaluate_deal_score import main as evaluate_main
        evaluate_main()
    finally:
        sys.argv = old_argv
    return 0


def run_final_buy_filter(args: argparse.Namespace) -> int:
    base = search_dir(args.folder)
    input_path = Path(args.input) if args.input else base / args.pipeline_out_dir / 'deals_ranked.csv'
    out_dir = Path(args.out_dir) if args.out_dir else base / args.final_buy_out_dir
    argv = [
        'final_buy_filter.py',
        '--input', str(input_path),
        '--out_dir', str(out_dir),
        '--top_n', str(args.top_n),
        '--max_workers', str(args.max_workers),
        '--min_resale_safety', str(args.min_resale_safety),
        '--min_deal_confidence', str(args.min_deal_confidence),
        '--min_expected_profit', str(args.min_expected_profit),
        '--min_expected_profit_margin', str(args.min_expected_profit_margin),
        '--min_buy_score', str(args.min_buy_score),
        '--min_seller_score', str(args.min_seller_score),
    ]
    if args.require_deal_eligible:
        argv.append('--require_deal_eligible')
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        from analysis_pipeline.scoring.final_buy_filter import main as filter_main
        filter_main()
    finally:
        sys.argv = old_argv
    return 0


def run_evaluate_buy_decisions(args: argparse.Namespace) -> int:
    base = search_dir(args.folder)
    input_path = Path(args.input) if args.input else base / args.final_buy_out_dir / 'buy_candidates_enriched.csv'
    sold = Path(args.sold) if args.sold else base / 'sold_df.csv'
    sold_eventually = None
    if args.sold_eventually:
        sold_eventually = Path(args.sold_eventually)
    elif args.use_eventual_sales:
        sold_eventually = base / args.eventual_out_dir / 'sold_eventually.csv'
    out_dir = Path(args.out_dir) if args.out_dir else base / args.buy_eval_out_dir
    argv = [
        'evaluate_buy_decisions.py',
        '--input', str(input_path),
        '--sold', str(sold),
        '--out_dir', str(out_dir),
        '--buy_score_threshold', str(args.buy_score_threshold),
    ]
    if sold_eventually is not None:
        argv.extend(['--sold_eventually', str(sold_eventually)])
    if args.no_dedupe:
        argv.append('--no_dedupe')
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        from analysis_pipeline.evaluation.evaluate_buy_decisions import main as evaluate_buy_main
        evaluate_buy_main()
    finally:
        sys.argv = old_argv
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Simple Python runner for scrape, batch, incremental, sold-eventually, evaluation, and final buy-filter workflows.')
    sub = ap.add_subparsers(dest='command', required=True)

    sub.add_parser('main', help='Run scripts/main.py').set_defaults(func=run_main_command)

    upd = sub.add_parser('update-eventual-sales', help='Run sold-eventually checks with optional candidate filters')
    upd.add_argument('--folder', required=True)
    upd.add_argument('--input', default=None)
    upd.add_argument('--out_dir', default=None)
    upd.add_argument('--eventual_out_dir', default='eventual_sale_check')
    upd.add_argument('--use_pipeline_out', action='store_true', help='Default input becomes pipeline_out/deals_ranked.csv instead of big_raw.csv')
    upd.add_argument('--max_workers', type=int, default=1)
    upd.add_argument('--delay', type=float, default=0.0)
    upd.add_argument('--allow_residential_fallback', action='store_true')
    upd.add_argument('--no_residential', action='store_true')
    upd.add_argument('--initial_delay', type=float, default=0.0)
    upd.add_argument('--fetch_sleep', type=float, default=60.0)
    upd.add_argument('--fetch_max_attempts', type=int, default=1)
    upd.add_argument('--recheck_sold_rows', action='store_true')
    upd.add_argument('--exclude_known_sold_csv', default=None)
    upd.add_argument('--min_deal_score', type=float, default=None)
    upd.add_argument('--min_deal_confidence', type=float, default=None)
    upd.add_argument('--top_n', type=int, default=None)
    upd.add_argument('--require_deal_eligible', action='store_true')
    upd.add_argument('--sort_by', default='DealScore,DealConfidence,SearchCount')
    upd.set_defaults(func=run_update_eventual_sales)

    batch = sub.add_parser('batch', help='Run the legacy clustering/scoring pipeline for one folder')
    batch.add_argument('--folder', required=True)
    batch.add_argument('--input', default=None)
    batch.add_argument('--input_name', default='big_raw.csv')
    batch.add_argument('--out_dir', default=None)
    batch.add_argument('--pipeline_out_dir', default='pipeline_out')
    batch.add_argument('--db', default=None)
    batch.add_argument('--model', default='paraphrase-multilingual-MiniLM-L12-v2')
    batch.add_argument('--product_threshold', type=float, default=0.24)
    batch.add_argument('--use_image_embeddings', action='store_true')
    batch.add_argument('--image_embedding_model', default='facebook/dinov2-base')
    batch.add_argument('--image_embedding_timeout', type=float, default=8.0)
    batch.add_argument('--image_embedding_batch_size', type=int, default=8)
    batch.add_argument('--product_image_weight', type=float, default=0.15)
    batch.add_argument('--autotune_variants', action='store_true')
    batch.add_argument('--variant_threshold', type=float, default=0.33)
    batch.add_argument('--core_frac', type=float, default=0.70)
    batch.add_argument('--variant_price_weight', type=float, default=0.35)
    batch.add_argument('--variant_image_weight', type=float, default=0.20)
    batch.add_argument('--min_product_size_for_variants', type=int, default=4)
    batch.add_argument('--min_variant_size_for_deals', type=int, default=3)
    batch.add_argument('--min_variant_size_for_confident_deals', type=int, default=3)
    batch.add_argument('--min_variant_silhouette', type=float, default=0.15)
    batch.add_argument('--max_variant_mad_ratio', type=float, default=0.35)
    batch.add_argument('--hard_max_variant_mad_ratio', type=float, default=0.60)
    batch.add_argument('--min_deal_confidence', type=float, default=0.35)
    batch.add_argument('--min_centroid_similarity', type=float, default=0.45)
    batch.add_argument('--min_informative_tokens', type=int, default=2)
    batch.add_argument('--exclude_negflag_deals', action='store_true')
    batch.add_argument('--resale_fee_rate', type=float, default=0.10)
    batch.add_argument('--resale_fixed_cost', type=float, default=0.0)
    batch.add_argument('--resale_safety_discount', type=float, default=0.05)
    batch.add_argument('--min_expected_profit', type=float, default=0.0)
    batch.add_argument('--min_expected_profit_margin', type=float, default=0.0)
    batch.add_argument('--price_buffer_size', type=int, default=200)
    batch.add_argument('--make_plots', action='store_true')
    batch.set_defaults(func=run_batch_command)

    inc = sub.add_parser('incremental', help='Run the incremental pipeline for new items only')
    inc.add_argument('--folder', required=True)
    inc.add_argument('--new_items', default=None)
    inc.add_argument('--growing_csv', default=None)
    inc.add_argument('--out_dir', default=None)
    inc.add_argument('--incremental_out_name', default='stream_assigned.csv')
    inc.add_argument('--pipeline_out_dir', default='pipeline_out')
    inc.add_argument('--db', default=None)
    inc.add_argument('--append', action='store_true')
    inc.add_argument('--price_buffer_size', type=int, default=200)
    inc.add_argument('--min_variant_size_for_confident_deals', type=int, default=3)
    inc.add_argument('--max_variant_mad_ratio', type=float, default=0.35)
    inc.add_argument('--hard_max_variant_mad_ratio', type=float, default=0.60)
    inc.add_argument('--min_deal_confidence', type=float, default=0.35)
    inc.add_argument('--min_centroid_similarity', type=float, default=0.45)
    inc.add_argument('--min_informative_tokens', type=int, default=2)
    inc.add_argument('--exclude_negflag_deals', action='store_true')
    inc.add_argument('--resale_fee_rate', type=float, default=0.10)
    inc.add_argument('--resale_fixed_cost', type=float, default=0.0)
    inc.add_argument('--resale_safety_discount', type=float, default=0.05)
    inc.add_argument('--min_expected_profit', type=float, default=0.0)
    inc.add_argument('--min_expected_profit_margin', type=float, default=0.0)
    inc.set_defaults(func=run_incremental_command)

    ev = sub.add_parser('evaluate', help='Run evaluate_deal_score.py for one folder')
    ev.add_argument('--folder', required=True)
    ev.add_argument('--deals', default=None)
    ev.add_argument('--sold', default=None)
    ev.add_argument('--sold_eventually', default=None)
    ev.add_argument('--use_eventual_sales', action='store_true')
    ev.add_argument('--out_dir', default=None)
    ev.add_argument('--pipeline_out_dir', default='pipeline_out')
    ev.add_argument('--eventual_out_dir', default='eventual_sale_check')
    ev.add_argument('--eval_out_dir', default='deal_score_eval')
    ev.add_argument('--high_score_threshold', type=float, default=2.0)
    ev.add_argument('--low_score_threshold', type=float, default=-1.0)
    ev.add_argument('--no_dedupe', action='store_true')
    ev.set_defaults(func=run_evaluate_command)

    fb = sub.add_parser('final-buy-filter', help='Enrich top candidates with full_scraper data and compute a final worth-buying decision')
    fb.add_argument('--folder', required=True)
    fb.add_argument('--input', default=None)
    fb.add_argument('--out_dir', default=None)
    fb.add_argument('--pipeline_out_dir', default='pipeline_out')
    fb.add_argument('--final_buy_out_dir', default='final_buy_filter')
    fb.add_argument('--top_n', type=int, default=30)
    fb.add_argument('--max_workers', type=int, default=4)
    fb.add_argument('--min_resale_safety', type=float, default=35.0)
    fb.add_argument('--min_deal_confidence', type=float, default=0.60)
    fb.add_argument('--min_expected_profit', type=float, default=8.0)
    fb.add_argument('--min_expected_profit_margin', type=float, default=0.12)
    fb.add_argument('--min_buy_score', type=float, default=0.62)
    fb.add_argument('--min_seller_score', type=float, default=0.45)
    fb.add_argument('--require_deal_eligible', action='store_true')
    fb.set_defaults(func=run_final_buy_filter)

    eb = sub.add_parser('evaluate-buy-decisions', help='Evaluate buy/not-buy outputs against sold and eventual-sold labels')
    eb.add_argument('--folder', required=True)
    eb.add_argument('--input', default=None)
    eb.add_argument('--sold', default=None)
    eb.add_argument('--sold_eventually', default=None)
    eb.add_argument('--use_eventual_sales', action='store_true')
    eb.add_argument('--out_dir', default=None)
    eb.add_argument('--final_buy_out_dir', default='final_buy_filter')
    eb.add_argument('--eventual_out_dir', default='eventual_sale_check')
    eb.add_argument('--buy_eval_out_dir', default='buy_eval')
    eb.add_argument('--buy_score_threshold', type=float, default=0.62)
    eb.add_argument('--no_dedupe', action='store_true')
    eb.set_defaults(func=run_evaluate_buy_decisions)

    return ap


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == '__main__':
    raise SystemExit(main())
