import json
from pathlib import Path

nodes = []
edges = []
hyperedges = []

def add_node(id, label, file_type, source_file, rationale=None):
    nodes.append({
        "id": id,
        "label": label,
        "file_type": file_type,
        "source_file": source_file,
        "source_location": None,
        "source_url": None,
        "captured_at": None,
        "author": None,
        "contributor": None,
        **({"rationale": rationale} if rationale else {}),
    })

def add_edge(source, target, relation, confidence, confidence_score, source_file):
    edges.append({
        "source": source,
        "target": target,
        "relation": relation,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "source_file": source_file,
        "source_location": None,
        "weight": 1.0,
    })

# Nodes
add_node("scripts_daily_eventual_sales_today_local_iso", "today_local_iso", "code", "scripts/daily_eventual_sales.py")
add_node("scripts_daily_eventual_sales_enqueue_priority_background_checks", "enqueue_priority_background_checks", "code", "scripts/daily_eventual_sales.py")
add_node("scripts_daily_eventual_sales_ensure_daily_deals_ranked", "ensure_daily_deals_ranked", "code", "scripts/daily_eventual_sales.py")
add_node("scripts_daily_eventual_sales_process_one_background_eventual_sale", "process_one_background_eventual_sale", "code", "scripts/daily_eventual_sales.py")
add_node("scripts_daily_eventual_sales_start_eventual_sales_background_worker", "start_eventual_sales_background_worker", "code", "scripts/daily_eventual_sales.py")
add_node("scripts_daily_eventual_sales_refresh_daily_eventual_sales", "refresh_daily_eventual_sales", "code", "scripts/daily_eventual_sales.py")
add_node("scripts_daily_eventual_sales_sync_priority_result_to_tracking_files", "_sync_priority_result_to_tracking_files", "code", "scripts/daily_eventual_sales.py")

add_node("scripts_filter_items_removerowscontainingwrongwords", "removeRowsContainingWrongWords", "code", "scripts/filter_items.py")
add_node("scripts_filter_items_filterout_wrongwordsintitle", "filterOut_WrongWordsInTitle", "code", "scripts/filter_items.py")
add_node("scripts_filter_items_apply_all_filters", "apply_all_filters", "code", "scripts/filter_items.py")

add_node("scripts_filters_find_brand_ids", "find_brand_ids", "code", "scripts/filters.py")
add_node("scripts_filters_find_color_ids", "find_color_ids", "code", "scripts/filters.py")

add_node("scripts_full_scrape_sold_history_run", "run", "code", "scripts/full_scrape_sold_history.py")
add_node("scripts_full_scrape_sold_history_pending_rows_for_search", "pending_rows_for_search", "code", "scripts/full_scrape_sold_history.py")

add_node("full_scraper_scrape_worker", "scrape_worker", "code", "scripts/full_scraper.py")
add_node("full_scraper_full_scraper", "Full_Scraper", "code", "scripts/full_scraper.py")
add_node("full_scraper_collect_and_store_full_items", "collect_and_store_full_items", "code", "scripts/full_scraper.py")
add_node("full_scraper_score_live_rows_for_search", "score_live_rows_for_search", "code", "scripts/full_scraper.py")
add_node("full_scraper_score_and_collect_extremes_for_live_rows", "score_and_collect_extremes_for_live_rows", "code", "scripts/full_scraper.py")
add_node("full_scraper_scrape_single_product", "scrape_single_product", "code", "scripts/full_scraper.py")
add_node("full_scraper_remove_not_actually_sold_items", "remove_not_actually_sold_items", "code", "scripts/full_scraper.py")

add_node("scripts_main_main", "main", "code", "scripts/main.py")

add_node("scripts_repair_big_raw_prices_repairdecision", "RepairDecision", "code", "scripts/repair_big_raw_prices.py")
add_node("scripts_repair_big_raw_prices_decide_price_repair", "decide_price_repair", "code", "scripts/repair_big_raw_prices.py")
add_node("scripts_repair_big_raw_prices_main", "main", "code", "scripts/repair_big_raw_prices.py")

add_node("scripts_report_download_stats_main", "main", "code", "scripts/report_download_stats.py")
add_node("scripts_report_proxy_identities_main", "main", "code", "scripts/report_proxy_identities.py")

add_node("scripts_save_context_snapshot_main", "main", "code", "scripts/save_context_snapshot.py")
add_node("scripts_save_context_snapshot_build_snapshot", "build_snapshot", "code", "scripts/save_context_snapshot.py")

add_node("scripts_scraping_options_preflight_parallel_scrape", "preflight_parallel_scrape", "code", "scripts/scraping_options.py")
add_node("scripts_scraping_options_dedupe_market_rows", "dedupe_market_rows", "code", "scripts/scraping_options.py")
add_node("scripts_scraping_options_write_csv_atomic", "write_csv_atomic", "code", "scripts/scraping_options.py")
add_node("scripts_scraping_options_ensure_search_tracking_files", "ensure_search_tracking_files", "code", "scripts/scraping_options.py")
add_node("scripts_scraping_options_compute_search_activity", "compute_search_activity", "code", "scripts/scraping_options.py")
add_node("scripts_scraping_options_build_search_run_plan", "build_search_run_plan", "code", "scripts/scraping_options.py")
add_node("scripts_scraping_options_scrape_specific_items_parallel", "scrape_specific_items_parallel", "code", "scripts/scraping_options.py")
add_node("scripts_scraping_options_update_eventual_sale_labels_for_csv", "update_eventual_sale_labels_for_csv", "code", "scripts/scraping_options.py")
add_node("scripts_scraping_options_parse_relative_upload_date_to_days", "parse_relative_upload_date_to_days", "code", "scripts/scraping_options.py")
add_node("scripts_scraping_options_read_schedule_state", "read_schedule_state", "code", "scripts/scraping_options.py")
add_node("scripts_scraping_options_update_search_schedule_state", "update_search_schedule_state", "code", "scripts/scraping_options.py")
add_node("scripts_scraping_options_filter_eventual_sale_candidate_rows", "filter_eventual_sale_candidate_rows", "code", "scripts/scraping_options.py")
add_node("scripts_scraping_options_update_market_status_for_df", "_update_market_status_for_df", "code", "scripts/scraping_options.py")

add_node("simple_scraper_simple_scraper", "Simple_scraper", "code", "scripts/simple_scraper.py")
add_node("simple_scraper_scrape_products_serial", "scrape_products_serial", "code", "scripts/simple_scraper.py")
add_node("simple_scraper_compare_and_save_df_serial", "compare_and_save_df_serial", "code", "scripts/simple_scraper.py")
add_node("simple_scraper_remove_not_actually_sold_items", "remove_not_actually_sold_items", "code", "scripts/simple_scraper.py")

add_node("scripts_stage_balanced_full_scrape_run", "run", "code", "scripts/stage_balanced_full_scrape.py")
add_node("scripts_stage_balanced_full_scrape_process_batches", "process_batches", "code", "scripts/stage_balanced_full_scrape.py")

add_node("scripts_workflow_runner_run_main_command", "run_main_command", "code", "scripts/workflow_runner.py")
add_node("scripts_workflow_runner_run_batch_command", "run_batch_command", "code", "scripts/workflow_runner.py")
add_node("scripts_workflow_runner_run_evaluate_command", "run_evaluate_command", "code", "scripts/workflow_runner.py")
add_node("scripts_workflow_runner_run_final_buy_filter", "run_final_buy_filter", "code", "scripts/workflow_runner.py")
add_node("scripts_workflow_runner_run_evaluate_buy_decisions", "run_evaluate_buy_decisions", "code", "scripts/workflow_runner.py")
add_node("scripts_workflow_runner_main", "main", "code", "scripts/workflow_runner.py")

add_node("evaluation_analyze_blur_signal_analyze_dataset", "analyze_dataset", "code", "scripts/analysis_pipeline/evaluation/analyze_blur_signal.py")
add_node("evaluation_analyze_blur_signal_summarize_signal", "summarize_signal", "code", "scripts/analysis_pipeline/evaluation/analyze_blur_signal.py")
add_node("evaluation_analyze_blur_signal_main", "main", "code", "scripts/analysis_pipeline/evaluation/analyze_blur_signal.py")

add_node("evaluation_analyze_clip_blur_signal_run_analysis", "run_analysis", "code", "scripts/analysis_pipeline/evaluation/analyze_clip_blur_signal.py")
add_node("evaluation_analyze_clip_blur_signal_main", "main", "code", "scripts/analysis_pipeline/evaluation/analyze_clip_blur_signal.py")

add_node("evaluation_analyze_foreground_blur_signal_main", "main", "code", "scripts/analysis_pipeline/evaluation/analyze_foreground_blur_signal.py")

add_node("evaluation_analyze_sold_funnel_main", "main", "code", "scripts/analysis_pipeline/evaluation/analyze_sold_funnel.py")

add_node("evaluation_build_balanced_buy_eval_dataset_main", "main", "code", "scripts/analysis_pipeline/evaluation/build_balanced_buy_eval_dataset.py")
add_node("evaluation_build_balanced_raw_dataset_main", "main", "code", "scripts/analysis_pipeline/evaluation/build_balanced_raw_dataset.py")

add_node("evaluation_cache_listing_images_main", "main", "code", "scripts/analysis_pipeline/evaluation/cache_listing_images.py")

add_node("evaluation_evaluate_buy_decisions_add_sold_labels", "add_sold_labels", "code", "scripts/analysis_pipeline/evaluation/evaluate_buy_decisions.py")
add_node("evaluation_evaluate_buy_decisions_build_buy_report", "build_buy_report", "code", "scripts/analysis_pipeline/evaluation/evaluate_buy_decisions.py")
add_node("evaluation_evaluate_buy_decisions_main", "main", "code", "scripts/analysis_pipeline/evaluation/evaluate_buy_decisions.py")

add_node("evaluation_evaluate_deal_score_dedupe_listings", "dedupe_listings", "code", "scripts/analysis_pipeline/evaluation/evaluate_deal_score.py")
add_node("evaluation_evaluate_deal_score_precision_at_k", "precision_at_k", "code", "scripts/analysis_pipeline/evaluation/evaluate_deal_score.py")
add_node("evaluation_evaluate_deal_score_precision_above_thresholds", "precision_above_thresholds", "code", "scripts/analysis_pipeline/evaluation/evaluate_deal_score.py")
add_node("evaluation_evaluate_deal_score_main", "main", "code", "scripts/analysis_pipeline/evaluation/evaluate_deal_score.py")

add_node("evaluation_prepare_balanced_visual_eval_input_main", "main", "code", "scripts/analysis_pipeline/evaluation/prepare_balanced_visual_eval_input.py")

add_node("evaluation_run_upstream_sweep_main", "main", "code", "scripts/analysis_pipeline/evaluation/run_upstream_sweep.py")

add_node("evaluation_tune_buy_pipeline_scan_deal_rules", "scan_deal_rules", "code", "scripts/analysis_pipeline/evaluation/tune_buy_pipeline.py")
add_node("evaluation_tune_buy_pipeline_scan_buy_rules", "scan_buy_rules", "code", "scripts/analysis_pipeline/evaluation/tune_buy_pipeline.py")
add_node("evaluation_tune_buy_pipeline_main", "main", "code", "scripts/analysis_pipeline/evaluation/tune_buy_pipeline.py")

add_node("evaluation_tune_full_enrichment_buy_policy_random_search", "random_search", "code", "scripts/analysis_pipeline/evaluation/tune_full_enrichment_buy_policy.py")
add_node("evaluation_tune_full_enrichment_buy_policy_main", "main", "code", "scripts/analysis_pipeline/evaluation/tune_full_enrichment_buy_policy.py")

add_node("evaluation_update_eventual_sales_main", "main", "code", "scripts/analysis_pipeline/evaluation/update_eventual_sales.py")

add_node("scoring_final_buy_filter_extract_buy_components", "extract_buy_components", "code", "scripts/analysis_pipeline/scoring/final_buy_filter.py")
add_node("scoring_final_buy_filter_compute_buy_decision", "compute_buy_decision", "code", "scripts/analysis_pipeline/scoring/final_buy_filter.py")
add_node("scoring_final_buy_filter_enrich_candidates", "enrich_candidates", "code", "scripts/analysis_pipeline/scoring/final_buy_filter.py")
add_node("scoring_final_buy_filter_enrich_one", "enrich_one", "code", "scripts/analysis_pipeline/scoring/final_buy_filter.py")
add_node("scoring_final_buy_filter_apply_visual_rerank", "apply_visual_rerank", "code", "scripts/analysis_pipeline/scoring/final_buy_filter.py")
add_node("scoring_final_buy_filter_select_candidates", "select_candidates", "code", "scripts/analysis_pipeline/scoring/final_buy_filter.py")
add_node("scoring_final_buy_filter_main", "main", "code", "scripts/analysis_pipeline/scoring/final_buy_filter.py",
         rationale="Generate final buy/not-buy decisions for deal candidates. Enriches listings with extra item/seller data, then computes BuyDecisionScore and WorthBuying.")

add_node("scoring_foreground_blur_foregroundblurmetrics", "ForegroundBlurMetrics", "code", "scripts/analysis_pipeline/scoring/foreground_blur.py")
add_node("scoring_foreground_blur_compute_foreground_blur_metrics", "compute_foreground_blur_metrics", "code", "scripts/analysis_pipeline/scoring/foreground_blur.py")

add_node("scoring_rerank_with_visuals_main", "main", "code", "scripts/analysis_pipeline/scoring/rerank_with_visuals.py")

# EXTRACTED edges
add_edge("scripts_daily_eventual_sales_enqueue_priority_background_checks", "scripts_scraping_options_dedupe_market_rows", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_enqueue_priority_background_checks", "scripts_scraping_options_write_csv_atomic", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_ensure_daily_deals_ranked", "scripts_scraping_options_read_schedule_state", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_ensure_daily_deals_ranked", "scripts_scraping_options_write_schedule_state", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_ensure_daily_deals_ranked", "scripts_workflow_runner_run_batch_command", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_process_one_background_eventual_sale", "scripts_scraping_options_update_market_status_for_df", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_process_one_background_eventual_sale", "scripts_scraping_options_write_schedule_state", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_process_one_background_eventual_sale", "scripts_daily_eventual_sales_ensure_daily_deals_ranked", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_process_one_background_eventual_sale", "scripts_daily_eventual_sales_sync_priority_result_to_tracking_files", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_sync_priority_result_to_tracking_files", "scripts_scraping_options_ensure_search_tracking_files", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_sync_priority_result_to_tracking_files", "scripts_scraping_options_dedupe_market_rows", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_sync_priority_result_to_tracking_files", "scripts_scraping_options_write_csv_atomic", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_sync_priority_result_to_tracking_files", "full_scraper_collect_and_store_full_items", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_refresh_daily_eventual_sales", "scripts_scraping_options_read_schedule_state", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_refresh_daily_eventual_sales", "scripts_scraping_options_update_eventual_sale_labels_for_csv", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_refresh_daily_eventual_sales", "scripts_scraping_options_write_schedule_state", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_refresh_daily_eventual_sales", "scripts_daily_eventual_sales_ensure_daily_deals_ranked", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")
add_edge("scripts_daily_eventual_sales_start_eventual_sales_background_worker", "scripts_daily_eventual_sales_process_one_background_eventual_sale", "calls", "EXTRACTED", 1.0, "scripts/daily_eventual_sales.py")

add_edge("scripts_main_main", "scripts_daily_eventual_sales_start_eventual_sales_background_worker", "calls", "EXTRACTED", 1.0, "scripts/main.py")
add_edge("scripts_main_main", "scripts_scraping_options_preflight_parallel_scrape", "calls", "EXTRACTED", 1.0, "scripts/main.py")
add_edge("scripts_main_main", "scripts_scraping_options_scrape_specific_items_parallel", "calls", "EXTRACTED", 1.0, "scripts/main.py")

add_edge("scripts_scraping_options_scrape_specific_items_parallel", "scripts_scraping_options_preflight_parallel_scrape", "calls", "EXTRACTED", 1.0, "scripts/scraping_options.py")
add_edge("scripts_scraping_options_scrape_specific_items_parallel", "scripts_scraping_options_build_search_run_plan", "calls", "EXTRACTED", 1.0, "scripts/scraping_options.py")
add_edge("scripts_scraping_options_scrape_specific_items_parallel", "scripts_scraping_options_compute_search_activity", "calls", "EXTRACTED", 1.0, "scripts/scraping_options.py")
add_edge("scripts_scraping_options_scrape_specific_items_parallel", "scripts_scraping_options_update_search_schedule_state", "calls", "EXTRACTED", 1.0, "scripts/scraping_options.py")
add_edge("scripts_scraping_options_build_search_run_plan", "scripts_scraping_options_ensure_search_tracking_files", "calls", "EXTRACTED", 1.0, "scripts/scraping_options.py")
add_edge("scripts_scraping_options_build_search_run_plan", "scripts_scraping_options_compute_search_activity", "calls", "EXTRACTED", 1.0, "scripts/scraping_options.py")
add_edge("scripts_scraping_options_build_search_run_plan", "scripts_scraping_options_read_schedule_state", "calls", "EXTRACTED", 1.0, "scripts/scraping_options.py")
add_edge("scripts_scraping_options_maybe_refresh_daily_eventual_sales_for_running_scheduler", "scripts_daily_eventual_sales_refresh_daily_eventual_sales", "calls", "EXTRACTED", 1.0, "scripts/scraping_options.py")
add_edge("scripts_scraping_options_maybe_refresh_daily_eventual_sales_for_running_scheduler", "scripts_daily_eventual_sales_today_local_iso", "calls", "EXTRACTED", 1.0, "scripts/scraping_options.py")
add_edge("scripts_scraping_options_update_eventual_sale_labels_for_csv", "scripts_scraping_options_dedupe_market_rows", "calls", "EXTRACTED", 1.0, "scripts/scraping_options.py")
add_edge("scripts_scraping_options_update_eventual_sale_labels_for_csv", "scripts_scraping_options_filter_eventual_sale_candidate_rows", "calls", "EXTRACTED", 1.0, "scripts/scraping_options.py")
add_edge("scripts_scraping_options_update_eventual_sale_labels_for_csv", "scripts_scraping_options_update_market_status_for_df", "calls", "EXTRACTED", 1.0, "scripts/scraping_options.py")

add_edge("simple_scraper_remove_not_actually_sold_items", "scripts_daily_eventual_sales_enqueue_priority_background_checks", "calls", "EXTRACTED", 1.0, "scripts/simple_scraper.py")
add_edge("simple_scraper_compare_and_save_df_serial", "simple_scraper_remove_not_actually_sold_items", "calls", "EXTRACTED", 1.0, "scripts/simple_scraper.py")

add_edge("full_scraper_scrape_products_serial", "full_scraper_scrape_worker", "calls", "EXTRACTED", 1.0, "scripts/full_scraper.py")
add_edge("full_scraper_score_and_collect_extremes_for_live_rows", "full_scraper_score_live_rows_for_search", "calls", "EXTRACTED", 1.0, "scripts/full_scraper.py")
add_edge("full_scraper_score_and_collect_extremes_for_live_rows", "full_scraper_collect_and_store_full_items", "calls", "EXTRACTED", 1.0, "scripts/full_scraper.py")

add_edge("scripts_full_scrape_sold_history_pending_rows_for_search", "full_scraper_full_scrape_paths", "calls", "EXTRACTED", 1.0, "scripts/full_scrape_sold_history.py")
add_edge("scripts_full_scrape_sold_history_pending_rows_for_search", "full_scraper_identity_keys_for_frame", "calls", "EXTRACTED", 1.0, "scripts/full_scrape_sold_history.py")
add_edge("scripts_full_scrape_sold_history_pending_rows_for_search", "full_scraper_identity_key", "calls", "EXTRACTED", 1.0, "scripts/full_scrape_sold_history.py")
add_edge("scripts_full_scrape_sold_history_run", "full_scraper_collect_and_store_full_items", "calls", "EXTRACTED", 1.0, "scripts/full_scrape_sold_history.py")

add_edge("scripts_stage_balanced_full_scrape_run", "scripts_stage_balanced_full_scrape_process_batches", "calls", "EXTRACTED", 1.0, "scripts/stage_balanced_full_scrape.py")
add_edge("scripts_stage_balanced_full_scrape_process_batches", "full_scraper_collect_and_store_full_items", "calls", "EXTRACTED", 1.0, "scripts/stage_balanced_full_scrape.py")

add_edge("scripts_workflow_runner_run_main_command", "scripts_main_main", "calls", "EXTRACTED", 1.0, "scripts/workflow_runner.py")
add_edge("scripts_workflow_runner_run_update_eventual_sales", "scripts_scraping_options_update_eventual_sale_labels_for_csv", "calls", "EXTRACTED", 1.0, "scripts/workflow_runner.py")
add_edge("scripts_workflow_runner_run_evaluate_command", "evaluation_evaluate_deal_score_main", "calls", "EXTRACTED", 1.0, "scripts/workflow_runner.py")
add_edge("scripts_workflow_runner_run_final_buy_filter", "scoring_final_buy_filter_main", "calls", "EXTRACTED", 1.0, "scripts/workflow_runner.py")
add_edge("scripts_workflow_runner_run_evaluate_buy_decisions", "evaluation_evaluate_buy_decisions_main", "calls", "EXTRACTED", 1.0, "scripts/workflow_runner.py")
add_edge("scripts_workflow_runner_main", "scripts_workflow_runner_run_main_command", "calls", "EXTRACTED", 1.0, "scripts/workflow_runner.py")
add_edge("scripts_workflow_runner_main", "scripts_workflow_runner_run_batch_command", "calls", "EXTRACTED", 1.0, "scripts/workflow_runner.py")
add_edge("scripts_workflow_runner_main", "scripts_workflow_runner_run_evaluate_command", "calls", "EXTRACTED", 1.0, "scripts/workflow_runner.py")
add_edge("scripts_workflow_runner_main", "scripts_workflow_runner_run_final_buy_filter", "calls", "EXTRACTED", 1.0, "scripts/workflow_runner.py")
add_edge("scripts_workflow_runner_main", "scripts_workflow_runner_run_evaluate_buy_decisions", "calls", "EXTRACTED", 1.0, "scripts/workflow_runner.py")

add_edge("evaluation_analyze_foreground_blur_signal_main", "scoring_foreground_blur_compute_foreground_blur_metrics", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/analyze_foreground_blur_signal.py")
add_edge("evaluation_analyze_foreground_blur_signal_main", "evaluation_analyze_blur_signal_bootstrap_mean_diff_ci", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/analyze_foreground_blur_signal.py")
add_edge("evaluation_analyze_foreground_blur_signal_main", "evaluation_analyze_blur_signal_cohens_d", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/analyze_foreground_blur_signal.py")
add_edge("evaluation_analyze_foreground_blur_signal_main", "evaluation_analyze_blur_signal_common_language_effect", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/analyze_foreground_blur_signal.py")
add_edge("evaluation_analyze_foreground_blur_signal_main", "evaluation_analyze_blur_signal_permutation_pvalue", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/analyze_foreground_blur_signal.py")

add_edge("evaluation_build_balanced_buy_eval_dataset_main", "evaluation_evaluate_buy_decisions_add_sold_labels", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/build_balanced_buy_eval_dataset.py")
add_edge("evaluation_build_balanced_raw_dataset_main", "evaluation_evaluate_deal_score_dedupe_listings", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/build_balanced_raw_dataset.py")

add_edge("evaluation_evaluate_buy_decisions_main", "evaluation_evaluate_deal_score_dedupe_listings", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/evaluate_buy_decisions.py")
add_edge("evaluation_evaluate_buy_decisions_main", "evaluation_evaluate_deal_score_precision_at_k", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/evaluate_buy_decisions.py")
add_edge("evaluation_evaluate_buy_decisions_main", "evaluation_evaluate_deal_score_precision_above_thresholds", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/evaluate_buy_decisions.py")

add_edge("evaluation_prepare_balanced_visual_eval_input_main", "evaluation_evaluate_deal_score_dedupe_listings", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/prepare_balanced_visual_eval_input.py")

add_edge("evaluation_run_upstream_sweep_main", "scoring_final_buy_filter_select_candidates", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/run_upstream_sweep.py")

add_edge("evaluation_tune_full_enrichment_buy_policy_main", "evaluation_evaluate_buy_decisions_add_sold_labels", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/tune_full_enrichment_buy_policy.py")
add_edge("evaluation_tune_full_enrichment_buy_policy_main", "scoring_final_buy_filter_extract_buy_components", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/tune_full_enrichment_buy_policy.py")

add_edge("evaluation_update_eventual_sales_main", "scripts_scraping_options_update_eventual_sale_labels_for_csv", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/update_eventual_sales.py")

add_edge("scoring_final_buy_filter_main", "scoring_final_buy_filter_select_candidates", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/scoring/final_buy_filter.py")
add_edge("scoring_final_buy_filter_main", "scoring_final_buy_filter_enrich_candidates", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/scoring/final_buy_filter.py")
add_edge("scoring_final_buy_filter_main", "scoring_final_buy_filter_apply_visual_rerank", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/scoring/final_buy_filter.py")
add_edge("scoring_final_buy_filter_main", "scoring_final_buy_filter_compute_buy_decision", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/scoring/final_buy_filter.py")
add_edge("scoring_final_buy_filter_enrich_candidates", "scoring_final_buy_filter_enrich_one", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/scoring/final_buy_filter.py")
add_edge("scoring_final_buy_filter_enrich_one", "full_scraper_scrape_single_product", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/scoring/final_buy_filter.py")

add_edge("scoring_rerank_with_visuals_main", "scoring_final_buy_filter_apply_visual_rerank", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/scoring/rerank_with_visuals.py")
add_edge("scoring_rerank_with_visuals_main", "scoring_final_buy_filter_compute_buy_decision", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/scoring/rerank_with_visuals.py")

add_edge("evaluation_analyze_blur_signal_main", "evaluation_analyze_blur_signal_analyze_dataset", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/analyze_blur_signal.py")
add_edge("evaluation_analyze_blur_signal_main", "evaluation_analyze_blur_signal_summarize_signal", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/analyze_blur_signal.py")
add_edge("evaluation_analyze_clip_blur_signal_main", "evaluation_analyze_clip_blur_signal_run_analysis", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/analyze_clip_blur_signal.py")
add_edge("evaluation_tune_buy_pipeline_main", "evaluation_tune_buy_pipeline_scan_deal_rules", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/tune_buy_pipeline.py")
add_edge("evaluation_tune_buy_pipeline_main", "evaluation_tune_buy_pipeline_scan_buy_rules", "calls", "EXTRACTED", 1.0, "scripts/analysis_pipeline/evaluation/tune_buy_pipeline.py")

# INFERRED edges
add_edge("scripts_scraping_options_dedupe_market_rows", "evaluation_evaluate_deal_score_dedupe_listings", "shares_data_with", "INFERRED", 0.85, "scripts/scraping_options.py")
add_edge("scripts_scraping_options_dedupe_market_rows", "full_scraper_full_scraper", "shares_data_with", "INFERRED", 0.75, "scripts/scraping_options.py")
add_edge("scripts_scraping_options_write_csv_atomic", "full_scraper_full_scraper", "shares_data_with", "INFERRED", 0.75, "scripts/scraping_options.py")
add_edge("simple_scraper_simple_scraper", "full_scraper_full_scraper", "semantically_similar_to", "INFERRED", 0.85, "scripts/simple_scraper.py")
add_edge("simple_scraper_remove_not_actually_sold_items", "full_scraper_remove_not_actually_sold_items", "semantically_similar_to", "INFERRED", 0.85, "scripts/simple_scraper.py")
add_edge("evaluation_build_balanced_buy_eval_dataset_main", "evaluation_build_balanced_raw_dataset_main", "semantically_similar_to", "INFERRED", 0.85, "scripts/analysis_pipeline/evaluation/build_balanced_buy_eval_dataset.py")
add_edge("evaluation_analyze_clip_blur_signal_run_analysis", "evaluation_analyze_blur_signal_analyze_dataset", "semantically_similar_to", "INFERRED", 0.85, "scripts/analysis_pipeline/evaluation/analyze_clip_blur_signal.py")
add_edge("scripts_full_scrape_sold_history_run", "scripts_stage_balanced_full_scrape_run", "semantically_similar_to", "INFERRED", 0.85, "scripts/full_scrape_sold_history.py")
add_edge("scripts_repair_big_raw_prices_main", "full_scraper_scrape_single_product", "conceptually_related_to", "INFERRED", 0.75, "scripts/repair_big_raw_prices.py")
add_edge("scripts_report_download_stats_main", "scripts_report_proxy_identities_main", "conceptually_related_to", "INFERRED", 0.75, "scripts/report_download_stats.py")
add_edge("scripts_filters_find_brand_ids", "scripts_filters_find_color_ids", "conceptually_related_to", "INFERRED", 0.85, "scripts/filters.py")
add_edge("evaluation_analyze_sold_funnel_main", "evaluation_evaluate_deal_score_main", "conceptually_related_to", "INFERRED", 0.85, "scripts/analysis_pipeline/evaluation/analyze_sold_funnel.py")
add_edge("evaluation_tune_buy_pipeline_main", "evaluation_tune_full_enrichment_buy_policy_main", "conceptually_related_to", "INFERRED", 0.85, "scripts/analysis_pipeline/evaluation/tune_buy_pipeline.py")
add_edge("evaluation_run_upstream_sweep_main", "evaluation_tune_buy_pipeline_main", "conceptually_related_to", "INFERRED", 0.85, "scripts/analysis_pipeline/evaluation/run_upstream_sweep.py")
add_edge("scoring_final_buy_filter_compute_buy_decision", "evaluation_evaluate_buy_decisions_build_buy_report", "shares_data_with", "INFERRED", 0.85, "scripts/analysis_pipeline/scoring/final_buy_filter.py")
add_edge("evaluation_evaluate_buy_decisions_build_buy_report", "evaluation_evaluate_deal_score_main", "conceptually_related_to", "INFERRED", 0.85, "scripts/analysis_pipeline/evaluation/evaluate_buy_decisions.py")

# Hyperedges
hyperedges.append({
    "id": "core_scraping_flow",
    "label": "Core Scraping Flow",
    "nodes": [
        "simple_scraper_simple_scraper",
        "full_scraper_full_scraper",
        "scripts_scraping_options_scrape_specific_items_parallel",
    ],
    "relation": "participate_in",
    "confidence": "INFERRED",
    "confidence_score": 0.85,
    "source_file": "scripts/scraping_options.py",
})
hyperedges.append({
    "id": "buy_decision_pipeline",
    "label": "Buy Decision Pipeline",
    "nodes": [
        "scoring_final_buy_filter_compute_buy_decision",
        "scoring_final_buy_filter_select_candidates",
        "evaluation_evaluate_buy_decisions_build_buy_report",
    ],
    "relation": "participate_in",
    "confidence": "INFERRED",
    "confidence_score": 0.85,
    "source_file": "scripts/analysis_pipeline/scoring/final_buy_filter.py",
})
hyperedges.append({
    "id": "blur_signal_analysis",
    "label": "Blur Signal Analysis",
    "nodes": [
        "evaluation_analyze_blur_signal_analyze_dataset",
        "evaluation_analyze_clip_blur_signal_run_analysis",
        "evaluation_analyze_foreground_blur_signal_main",
    ],
    "relation": "participate_in",
    "confidence": "INFERRED",
    "confidence_score": 0.85,
    "source_file": "scripts/analysis_pipeline/evaluation/analyze_blur_signal.py",
})

payload = {
    "nodes": nodes,
    "edges": edges,
    "hyperedges": hyperedges,
    "input_tokens": 0,
    "output_tokens": 0,
}

out_path = Path("/home/ale/Desktop/vinted/Vinted_New_Version/graphify-out/.graphify_chunk_01.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"Wrote {len(nodes)} nodes, {len(edges)} edges, {len(hyperedges)} hyperedges to {out_path}")
