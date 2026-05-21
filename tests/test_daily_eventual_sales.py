import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

sys.modules.setdefault('dotenv', types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))

fake_scraping_options = types.ModuleType('scraping_options')
fake_scraping_options.read_schedule_state = lambda *_args, **_kwargs: {}
fake_scraping_options.write_schedule_state = lambda *_args, **_kwargs: None
fake_scraping_options.update_eventual_sale_labels_for_csv = lambda *_args, **_kwargs: {}
fake_scraping_options.read_schedule_state = lambda *_args, **_kwargs: {}
fake_scraping_options.write_schedule_state = lambda *_args, **_kwargs: None
fake_scraping_options.filter_eventual_sale_candidate_rows = lambda df, **_kwargs: df.copy()
fake_scraping_options.dedupe_market_rows = lambda df, keep='last': df.drop_duplicates(subset=['Dataid'], keep=keep) if 'Dataid' in df.columns else df.copy()
fake_scraping_options.exclude_matching_market_rows = lambda df, reference_df: (
    df.loc[
        ~pd.to_numeric(df.get('Dataid', pd.Series(dtype='float64')), errors='coerce').isin(
            pd.to_numeric(reference_df.get('Dataid', pd.Series(dtype='float64')), errors='coerce').dropna().tolist()
        )
    ].copy()
    if 'Dataid' in df.columns and 'Dataid' in reference_df.columns
    else df.copy()
)
fake_scraping_options._update_market_status_for_df = lambda df, **_kwargs: (df.copy(), pd.DataFrame(columns=df.columns))
fake_scraping_options.write_csv_atomic = lambda df, path: df.to_csv(path, index=False)
fake_scraping_options.ensure_search_tracking_files = lambda output_folder: [Path(output_folder).mkdir(parents=True, exist_ok=True)] or None
sys.modules['scraping_options'] = fake_scraping_options

import daily_eventual_sales as daily_module


class DailyEventuallySoldTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.simple_scrape_dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_refresh_daily_eventual_sales_runs_and_updates_state(self):
        search = SimpleNamespace(folder='ps4')
        output_folder = self.simple_scrape_dir / search.folder
        output_folder.mkdir(parents=True, exist_ok=True)
        deals_path = output_folder / daily_module.PIPELINE_OUT_DIR_NAME / daily_module.DEALS_RANKED_FILENAME
        deals_path.parent.mkdir(parents=True, exist_ok=True)
        deals_path.write_text('Dataid\n1\n', encoding='utf-8')

        writes = []
        refresh_calls = []

        sys.modules['scraping_options'] = fake_scraping_options
        sys.modules['scraping_options'] = fake_scraping_options
        with patch.object(daily_module, 'settings', SimpleNamespace(paths=SimpleNamespace(simple_scrape_dir=self.simple_scrape_dir))):
            with patch.object(fake_scraping_options, 'read_schedule_state', return_value={}):
                with patch.object(fake_scraping_options, 'write_schedule_state', side_effect=lambda folder, state: writes.append((folder, state.copy()))):
                    with patch.object(daily_module, 'ensure_daily_deals_ranked', return_value=deals_path):
                        with patch.object(
                            fake_scraping_options, 'update_eventual_sale_labels_for_csv',
                            side_effect=lambda *args, **kwargs: refresh_calls.append((args, kwargs)) or {'n_checked': 5, 'n_sold': 2},
                        ):
                            daily_module.refresh_daily_eventual_sales([search], today_iso='2026-03-22')

        self.assertEqual(len(refresh_calls), 1)
        args, kwargs = refresh_calls[0]
        self.assertEqual(args[0], str(deals_path))
        self.assertEqual(kwargs['out_dir'], str(output_folder / daily_module.EVENTUAL_SALES_OUT_DIR_NAME))
        self.assertEqual(kwargs['max_workers'], daily_module.EVENTUAL_SALES_MAX_WORKERS)
        self.assertEqual(kwargs['delay'], daily_module.EVENTUAL_SALES_DELAY_SECONDS)
        self.assertNotIn('allow_residential_fallback', kwargs)
        self.assertNotIn('no_residential', kwargs)
        self.assertEqual(kwargs['fetch_sleep'], daily_module.EVENTUAL_SALES_FETCH_SLEEP_SECONDS)
        self.assertEqual(kwargs['fetch_max_attempts'], daily_module.EVENTUAL_SALES_FETCH_MAX_ATTEMPTS)
        self.assertEqual(kwargs['min_deal_score'], 2.0)
        self.assertEqual(kwargs['min_deal_confidence'], 0.7)
        self.assertEqual(kwargs['top_n'], 100)
        self.assertTrue(kwargs['require_deal_eligible'])
        self.assertEqual(kwargs['sort_by'], 'DealScore,DealConfidence,SearchCount')
        self.assertEqual(kwargs['exclude_known_sold_csv'], str(output_folder / 'sold_df.csv'))
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][1][daily_module.EVENTUAL_SALES_STATE_DATE_KEY], '2026-03-22')
        self.assertEqual(writes[0][1]['last_eventual_sale_refresh_checked'], 5)
        self.assertEqual(writes[0][1]['last_eventual_sale_refresh_sold'], 2)

    def test_refresh_daily_eventual_sales_skips_when_already_done_today(self):
        search = SimpleNamespace(folder='gucci')
        output_folder = self.simple_scrape_dir / search.folder
        output_folder.mkdir(parents=True, exist_ok=True)

        with patch.object(daily_module, 'settings', SimpleNamespace(paths=SimpleNamespace(simple_scrape_dir=self.simple_scrape_dir))):
            with patch.object(fake_scraping_options, 'read_schedule_state', return_value={daily_module.EVENTUAL_SALES_STATE_DATE_KEY: '2026-03-22'}):
                with patch.object(fake_scraping_options, 'update_eventual_sale_labels_for_csv') as refresh_mock:
                    with patch.object(fake_scraping_options, 'write_schedule_state') as write_mock:
                        daily_module.refresh_daily_eventual_sales([search], today_iso='2026-03-22')

        refresh_mock.assert_not_called()
        write_mock.assert_not_called()

    def test_process_one_background_eventual_sale_updates_outputs_and_state(self):
        search = SimpleNamespace(folder='prada')
        output_folder = self.simple_scrape_dir / search.folder
        output_folder.mkdir(parents=True, exist_ok=True)
        deals_path = output_folder / daily_module.PIPELINE_OUT_DIR_NAME / daily_module.DEALS_RANKED_FILENAME
        deals_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    'Dataid': 10,
                    'Link': 'https://www.vinted.it/items/10-test',
                    'DealScore': 5.0,
                    'DealConfidence': 0.9,
                    'DealEligible': True,
                    'MarketStatus': 'On Sale',
                }
            ]
        ).to_csv(deals_path, index=False)

        state_store = {}

        def read_state(_folder):
            return state_store.copy()

        def write_state(_folder, state):
            state_store.clear()
            state_store.update(state)

        def fake_update(df, **_kwargs):
            checked = df.copy()
            checked.loc[:, 'MarketStatus'] = 'Sold'
            checked.loc[:, 'ObservedPagePrice'] = 42.0
            return checked, checked.copy()

        with patch.dict(sys.modules, {'scraping_options': fake_scraping_options}):
            with patch.object(daily_module, 'settings', SimpleNamespace(paths=SimpleNamespace(simple_scrape_dir=self.simple_scrape_dir))):
                with patch.object(fake_scraping_options, 'read_schedule_state', side_effect=read_state):
                    with patch.object(fake_scraping_options, 'write_schedule_state', side_effect=write_state):
                        with patch.object(fake_scraping_options, '_update_market_status_for_df', side_effect=fake_update):
                            with patch.object(daily_module, 'ensure_daily_deals_ranked', return_value=deals_path):
                                sold = daily_module.process_one_background_eventual_sale(search)

        self.assertTrue(sold)
        sold_csv = output_folder / daily_module.EVENTUAL_SALES_OUT_DIR_NAME / 'sold_eventually.csv'
        labeled_csv = output_folder / daily_module.EVENTUAL_SALES_OUT_DIR_NAME / 'big_raw_eventual_sale_labeled.csv'
        self.assertTrue(sold_csv.exists())
        self.assertTrue(labeled_csv.exists())
        sold_df = pd.read_csv(sold_csv)
        self.assertEqual(len(sold_df), 1)
        self.assertEqual(int(sold_df.loc[0, 'Dataid']), 10)
        self.assertEqual(state_store[daily_module.EVENTUAL_SALES_BACKGROUND_LAST_DATAID_KEY], '10')
        self.assertEqual(state_store[daily_module.EVENTUAL_SALES_BACKGROUND_LAST_STATUS_KEY], 'Sold')

    def test_select_next_background_candidate_skips_items_already_in_sold_df(self):
        search = SimpleNamespace(folder='gucci')
        output_folder = self.simple_scrape_dir / search.folder
        output_folder.mkdir(parents=True, exist_ok=True)
        deals_path = output_folder / daily_module.PIPELINE_OUT_DIR_NAME / daily_module.DEALS_RANKED_FILENAME
        deals_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {'Dataid': 10, 'Link': 'https://www.vinted.it/items/10-test', 'DealEligible': True},
                {'Dataid': 11, 'Link': 'https://www.vinted.it/items/11-test', 'DealEligible': True},
            ]
        ).to_csv(deals_path, index=False)
        pd.DataFrame(
            [{'Dataid': 10, 'Link': 'https://www.vinted.it/items/10-test', 'MarketStatus': 'Sold'}]
        ).to_csv(output_folder / 'sold_df.csv', index=False)

        with patch.dict(sys.modules, {'scraping_options': fake_scraping_options}):
            with patch.object(daily_module, 'settings', SimpleNamespace(paths=SimpleNamespace(simple_scrape_dir=self.simple_scrape_dir))):
                with patch.object(fake_scraping_options, 'read_schedule_state', return_value={}):
                    source, input_path, candidate_df, state = daily_module._select_next_background_candidate(search)

        self.assertEqual(source, 'deals_ranked')
        self.assertEqual(input_path, deals_path)
        self.assertEqual(state, {})
        self.assertIsNotNone(candidate_df)
        self.assertEqual(int(candidate_df.iloc[0]['Dataid']), 11)

    def test_ensure_daily_deals_ranked_runs_batch_once_per_day(self):
        search = SimpleNamespace(folder='ps4')
        output_folder = self.simple_scrape_dir / search.folder
        output_folder.mkdir(parents=True, exist_ok=True)
        big_raw_path = output_folder / 'big_raw.csv'
        big_raw_path.write_text('Dataid\n1\n', encoding='utf-8')
        deals_path = output_folder / daily_module.PIPELINE_OUT_DIR_NAME / daily_module.DEALS_RANKED_FILENAME

        state_store = {}
        batch_calls = []

        def read_state(_folder):
            return state_store.copy()

        def write_state(_folder, state):
            state_store.clear()
            state_store.update(state)

        def fake_run_batch(args):
            batch_calls.append(args)
            deals_path.parent.mkdir(parents=True, exist_ok=True)
            deals_path.write_text('Dataid\n1\n', encoding='utf-8')
            return 0

        with patch.dict(sys.modules, {'scraping_options': fake_scraping_options}):
            with patch.object(daily_module, 'settings', SimpleNamespace(paths=SimpleNamespace(simple_scrape_dir=self.simple_scrape_dir))):
                with patch.object(fake_scraping_options, 'read_schedule_state', side_effect=read_state):
                    with patch.object(fake_scraping_options, 'write_schedule_state', side_effect=write_state):
                        with patch('workflow_runner.run_batch_command', side_effect=fake_run_batch):
                            first = daily_module.ensure_daily_deals_ranked(search, today_iso='2026-03-22')
                            second = daily_module.ensure_daily_deals_ranked(search, today_iso='2026-03-22')

        self.assertEqual(first, deals_path)
        self.assertEqual(second, deals_path)
        self.assertEqual(len(batch_calls), 1)
        self.assertEqual(batch_calls[0].input, str(big_raw_path))
        self.assertEqual(state_store[daily_module.DEALS_RANKED_REFRESH_DATE_KEY], '2026-03-22')
        self.assertEqual(state_store['last_deals_ranked_refresh_input'], 'big_raw.csv')

    def test_process_one_background_eventual_sale_prioritizes_queue(self):
        search = SimpleNamespace(folder='gucci')
        output_folder = self.simple_scrape_dir / search.folder
        queue_dir = output_folder / daily_module.EVENTUAL_SALES_OUT_DIR_NAME
        queue_dir.mkdir(parents=True, exist_ok=True)
        queue_path = queue_dir / daily_module.PRIORITY_CHECK_QUEUE_FILENAME
        pd.DataFrame(
            [
                {
                    'Dataid': 77,
                    'Link': 'https://www.vinted.it/items/77-test',
                    'Title': 'Priority item',
                    daily_module.PRIORITY_QUEUE_SOURCE_COLUMN: 'compare_and_save',
                    daily_module.PRIORITY_QUEUE_ENQUEUED_AT_COLUMN: '2026-05-08T10:00:00+02:00',
                    daily_module.PRIORITY_QUEUE_ATTEMPTS_COLUMN: 0,
                }
            ]
        ).to_csv(queue_path, index=False)

        state_store = {}

        def read_state(_folder):
            return state_store.copy()

        def write_state(_folder, state):
            state_store.clear()
            state_store.update(state)

        def ensure_tracking(output_folder_str):
            folder = Path(output_folder_str)
            folder.mkdir(parents=True, exist_ok=True)
            for filename in ('old_df.csv', 'unsold_df.csv', 'sold_df.csv', 'non_really_sold_items_ids.csv'):
                path = folder / filename
                if path.exists():
                    continue
                cols = ['Dataid'] if filename == 'non_really_sold_items_ids.csv' else ['Dataid', 'Link', 'Title', 'MarketStatus']
                pd.DataFrame(columns=cols).to_csv(path, index=False)

        def fake_update(df, **_kwargs):
            checked = df.copy()
            checked.loc[:, 'MarketStatus'] = 'On Sale'
            checked.loc[:, 'LastCheckStatus'] = 'On Sale'
            return checked, pd.DataFrame(columns=checked.columns)

        with patch.dict(sys.modules, {'scraping_options': fake_scraping_options}):
            with patch.object(daily_module, 'settings', SimpleNamespace(paths=SimpleNamespace(simple_scrape_dir=self.simple_scrape_dir))):
                with patch.object(fake_scraping_options, 'read_schedule_state', side_effect=read_state):
                    with patch.object(fake_scraping_options, 'write_schedule_state', side_effect=write_state):
                        with patch.object(fake_scraping_options, '_update_market_status_for_df', side_effect=fake_update):
                            with patch.object(fake_scraping_options, 'ensure_search_tracking_files', side_effect=ensure_tracking):
                                with patch.object(daily_module, 'ensure_daily_deals_ranked') as ensure_mock:
                                    sold = daily_module.process_one_background_eventual_sale(search)

        self.assertFalse(sold)
        ensure_mock.assert_not_called()
        remaining_queue = pd.read_csv(queue_path)
        self.assertEqual(len(remaining_queue), 0)
        old_df = pd.read_csv(output_folder / 'old_df.csv')
        self.assertEqual(old_df['Dataid'].tolist(), [77])
        non_really_df = pd.read_csv(output_folder / 'non_really_sold_items_ids.csv')
        self.assertEqual(non_really_df['Dataid'].tolist(), [77])
        self.assertEqual(state_store[daily_module.EVENTUAL_SALES_BACKGROUND_LAST_SOURCE_KEY], 'priority')

    def test_priority_sold_items_update_sold_df_without_entering_sold_eventually(self):
        search = SimpleNamespace(folder='borse')
        output_folder = self.simple_scrape_dir / search.folder
        queue_dir = output_folder / daily_module.EVENTUAL_SALES_OUT_DIR_NAME
        queue_dir.mkdir(parents=True, exist_ok=True)
        queue_path = queue_dir / daily_module.PRIORITY_CHECK_QUEUE_FILENAME
        pd.DataFrame(
            [
                {
                    'Dataid': 88,
                    'Link': 'https://www.vinted.it/items/88-test',
                    'Title': 'Priority sold item',
                    daily_module.PRIORITY_QUEUE_SOURCE_COLUMN: 'compare_and_save',
                    daily_module.PRIORITY_QUEUE_ENQUEUED_AT_COLUMN: '2026-05-08T10:00:00+02:00',
                    daily_module.PRIORITY_QUEUE_ATTEMPTS_COLUMN: 0,
                }
            ]
        ).to_csv(queue_path, index=False)

        state_store = {}

        def read_state(_folder):
            return state_store.copy()

        def write_state(_folder, state):
            state_store.clear()
            state_store.update(state)

        def ensure_tracking(output_folder_str):
            folder = Path(output_folder_str)
            folder.mkdir(parents=True, exist_ok=True)
            for filename in ('old_df.csv', 'unsold_df.csv', 'sold_df.csv', 'non_really_sold_items_ids.csv'):
                path = folder / filename
                if path.exists():
                    continue
                cols = ['Dataid'] if filename == 'non_really_sold_items_ids.csv' else ['Dataid', 'Link', 'Title', 'MarketStatus']
                pd.DataFrame(columns=cols).to_csv(path, index=False)

        def fake_update(df, **_kwargs):
            checked = df.copy()
            checked.loc[:, 'MarketStatus'] = 'Sold'
            checked.loc[:, 'LastCheckStatus'] = 'Sold'
            return checked, checked.copy()

        with patch.dict(sys.modules, {'scraping_options': fake_scraping_options}):
            with patch.object(daily_module, 'settings', SimpleNamespace(paths=SimpleNamespace(simple_scrape_dir=self.simple_scrape_dir))):
                with patch.object(fake_scraping_options, 'read_schedule_state', side_effect=read_state):
                    with patch.object(fake_scraping_options, 'write_schedule_state', side_effect=write_state):
                        with patch.object(fake_scraping_options, '_update_market_status_for_df', side_effect=fake_update):
                            with patch.object(fake_scraping_options, 'ensure_search_tracking_files', side_effect=ensure_tracking):
                                sold = daily_module.process_one_background_eventual_sale(search)

        self.assertTrue(sold)
        sold_df = pd.read_csv(output_folder / 'sold_df.csv')
        self.assertEqual(sold_df['Dataid'].astype(int).tolist(), [88])
        eventual_sold_path = output_folder / daily_module.EVENTUAL_SALES_OUT_DIR_NAME / 'sold_eventually.csv'
        eventual_sold_df = pd.read_csv(eventual_sold_path)
        self.assertEqual(len(eventual_sold_df), 0)

    def test_finalize_priority_candidate_requeues_fetchfailed_and_increments_attempts(self):
        output_folder = self.simple_scrape_dir / 'ps4'
        queue_dir = output_folder / daily_module.EVENTUAL_SALES_OUT_DIR_NAME
        queue_dir.mkdir(parents=True, exist_ok=True)
        queue_path = queue_dir / daily_module.PRIORITY_CHECK_QUEUE_FILENAME
        pd.DataFrame(
            [
                {
                    'Dataid': 77,
                    'Link': 'https://www.vinted.it/items/77-test',
                    daily_module.PRIORITY_QUEUE_SOURCE_COLUMN: 'compare_and_save',
                    daily_module.PRIORITY_QUEUE_ENQUEUED_AT_COLUMN: '2026-04-07T10:00:00+02:00',
                    daily_module.PRIORITY_QUEUE_LAST_STATUS_COLUMN: pd.NA,
                    daily_module.PRIORITY_QUEUE_LAST_CHECKED_AT_COLUMN: pd.NA,
                    daily_module.PRIORITY_QUEUE_ATTEMPTS_COLUMN: 0,
                }
            ]
        ).to_csv(queue_path, index=False)

        checked_row = pd.DataFrame(
            [
                {
                    'Dataid': 77,
                    'Link': 'https://www.vinted.it/items/77-test',
                    'LastCheckStatus': 'FetchFailed',
                }
            ]
        )

        with patch.dict(sys.modules, {'scraping_options': fake_scraping_options}):
            daily_module._finalize_priority_candidate(output_folder, checked_row)

        queue_df = pd.read_csv(queue_path)
        self.assertEqual(len(queue_df), 1)
        self.assertEqual(str(queue_df.loc[0, daily_module.PRIORITY_QUEUE_LAST_STATUS_COLUMN]), 'FetchFailed')
        self.assertEqual(int(queue_df.loc[0, daily_module.PRIORITY_QUEUE_ATTEMPTS_COLUMN]), 1)

    def test_select_next_priority_candidate_skips_rows_older_than_30_days(self):
        search = SimpleNamespace(folder='ps4')
        output_folder = self.simple_scrape_dir / search.folder
        queue_dir = output_folder / daily_module.EVENTUAL_SALES_OUT_DIR_NAME
        queue_dir.mkdir(parents=True, exist_ok=True)
        queue_path = queue_dir / daily_module.PRIORITY_CHECK_QUEUE_FILENAME
        pd.DataFrame(
            [
                {
                    'Dataid': 1,
                    'Link': 'https://www.vinted.it/items/1-old',
                    'SearchDate': '01/03/2026 10:00:00',
                    daily_module.PRIORITY_QUEUE_SOURCE_COLUMN: 'compare_and_save',
                    daily_module.PRIORITY_QUEUE_ENQUEUED_AT_COLUMN: '2026-03-01T10:00:00+01:00',
                    daily_module.PRIORITY_QUEUE_ATTEMPTS_COLUMN: 0,
                },
                {
                    'Dataid': 2,
                    'Link': 'https://www.vinted.it/items/2-recent',
                    'SearchDate': '08/05/2026 10:00:00',
                    daily_module.PRIORITY_QUEUE_SOURCE_COLUMN: 'compare_and_save',
                    daily_module.PRIORITY_QUEUE_ENQUEUED_AT_COLUMN: '2026-05-08T10:00:00+02:00',
                    daily_module.PRIORITY_QUEUE_ATTEMPTS_COLUMN: 0,
                },
            ]
        ).to_csv(queue_path, index=False)

        with patch.object(daily_module, 'settings', SimpleNamespace(paths=SimpleNamespace(simple_scrape_dir=self.simple_scrape_dir))):
            queue_path_out, row = daily_module._select_next_priority_candidate(search)

        self.assertEqual(queue_path_out, queue_path)
        self.assertIsNotNone(row)
        self.assertEqual(int(row.iloc[0]['Dataid']), 2)
        queue_df = pd.read_csv(queue_path)
        self.assertEqual(set(queue_df['Dataid'].astype(int).tolist()), {2})


if __name__ == '__main__':
    unittest.main()
