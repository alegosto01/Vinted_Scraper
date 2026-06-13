from Scraper import Scraper
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
from datetime import datetime
import multiprocessing
import requests_html
from pathlib import Path

from config.project_config import settings
from utils_lib.image_cache import download_listing_images_to_cache


MAX_OLD_ITEMS_PER_SEARCH = 1000


class Simple_scraper(Scraper):
    def __init__(self):
        super().__init__()

    def _dedupe_listing_rows(self, df):
        if df.empty:
            return df.copy()

        out = df.copy()
        if "Dataid" in out.columns:
            dataid_num = pd.to_numeric(out["Dataid"], errors="coerce")
            with_id = out.loc[dataid_num.notna()].copy()
            without_id = out.loc[dataid_num.isna()].copy()

            if not with_id.empty:
                with_id["_DataidNum"] = pd.to_numeric(with_id["Dataid"], errors="coerce").astype("Int64")
                with_id = with_id.drop_duplicates(subset=["_DataidNum"], keep="last").drop(columns=["_DataidNum"])

            if not without_id.empty and "Link" in without_id.columns:
                without_id = without_id.drop_duplicates(subset=["Link"], keep="last")

            out = pd.concat([with_id, without_id], ignore_index=True)
        elif "Link" in out.columns:
            out = out.drop_duplicates(subset=["Link"], keep="last")

        return out.reset_index(drop=True)

    def _write_csv_atomic(self, df, csv_path):
        tmp_path = f"{csv_path}.tmp"
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, csv_path)

    def _load_listing_id_cache(self, csv_path, id_column="Dataid"):
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                self.logger.warning('Failed reading id cache %s: %s', csv_path, exc)
                return pd.DataFrame(columns=[id_column])
            if id_column not in df.columns:
                df = pd.DataFrame(columns=[id_column])
        else:
            df = pd.DataFrame(columns=[id_column])

        if not df.empty:
            df[id_column] = pd.to_numeric(df[id_column], errors='coerce')
            df = df.dropna(subset=[id_column]).copy()
            df[id_column] = df[id_column].astype(int)
            df = df.drop_duplicates(subset=[id_column], keep='first')

        return df

    def _rank_rows_for_old_df_replacement(self, old_df):
        if old_df.empty:
            return old_df.copy()

        ranked = old_df.copy()
        ranked['_row_order'] = range(len(ranked))

        if 'SearchDate' in ranked.columns:
            ranked['_SearchDateTs'] = pd.to_datetime(ranked['SearchDate'], errors='coerce', dayfirst=True)
        else:
            ranked['_SearchDateTs'] = pd.Series(pd.NaT, index=ranked.index)

        ranked = ranked.sort_values(
            by=['_SearchDateTs', '_row_order'],
            ascending=[True, True],
            na_position='last',
            kind='stable',
        )
        return ranked

    def _title_matches_wrong_words(self, title, wrong_words):
        if not wrong_words or not title:
            return False
        title_lower = title.lower()
        return any(word.strip().lower() in title_lower for word in wrong_words if word.strip())

    def fetch_page_and_check(self, item, get_images=False, check_venduto=True, get_upload_date=False):
        return self.inspect_item_page(
            item,
            get_images=get_images,
            check_venduto=check_venduto,
            get_upload_date=get_upload_date,
            initial_delay=20,
        )

    def extract_product_meta(self, product, all_likes_counts, page, search_count, get_images):
        return self.extract_catalog_item_meta(
            product,
            all_likes_counts,
            page,
            search_count,
            get_images=get_images,
        )

    def _catalog_image_cache_root(self, search_folder: str) -> Path:
        return Path(str(settings.paths.simple_scrape_dir)) / str(search_folder) / "image_cache"

    def _attach_local_catalog_images(self, row: dict, search_folder: str) -> dict:
        row = dict(row)
        row.setdefault("LocalImagePaths", "[]")
        row.setdefault("LocalPrimaryImagePath", "")

        data_id = row.get("Dataid")
        if not data_id:
            return row

        image_urls = row.get("Images")
        if not image_urls:
            return row

        try:
            local_paths = download_listing_images_to_cache(
                image_urls,
                dataset_root=self._catalog_image_cache_root(search_folder),
                data_id=data_id,
                timeout=20.0,
                overwrite=False,
            )
        except Exception as exc:
            self.logger.warning('Catalog image download failed for item=%s: %s', data_id, exc)
            return row

        row["LocalImagePaths"] = json.dumps(local_paths)
        row["LocalPrimaryImagePath"] = local_paths[0] if local_paths else ""
        return row

    def scrape_products_serial(self, dictionary, search_count, pages_to_scrape, get_images=True):
        data = []
        stats = {
            'pages_attempted': 0,
            'pages_loaded': 0,
            'pages_parse_failed': 0,
            'products_seen': 0,
            'products_valid': 0,
            'products_dropped': 0,
            'products_excluded_wrong_words': 0,
        }
        webpage = self.create_webpage(dictionary)

        for page in range(pages_to_scrape):
            stats['pages_attempted'] += 1
            new_webpage = webpage + '&page=' + str(page + 1)
            self.logger.info('Simple scrape search=%s page=%s url=%s', dictionary.folder, page + 1, new_webpage)

            html_text = self.get_page_content_datacenter(new_webpage, request_kind='catalog_page')
            if not isinstance(html_text, str) or not html_text.strip():
                stats['pages_parse_failed'] += 1
                self.logger.warning('Skipping empty catalog response for search=%s page=%s', dictionary.folder, page + 1)
                continue

            try:
                html_content = requests_html.HTML(html=html_text)
                stats['pages_loaded'] += 1
            except Exception as exc:
                stats['pages_parse_failed'] += 1
                self.logger.warning('Catalog HTML parsing failed for search=%s page=%s: %s', dictionary.folder, page + 1, exc)
                try:
                    with open('debug_vinted_page.html', 'w', encoding='utf-8', errors='ignore') as handle:
                        handle.write(html_text)
                except OSError as file_exc:
                    self.logger.warning('Could not write debug catalog HTML for page=%s: %s', page + 1, file_exc)
                continue

            time.sleep(7)
            products = html_content.find('.new-item-box__container')
            all_likes_counts = html_content.find('.u-background-white.u-flexbox.u-align-items-center.new-item-box__favourite-icon')
            stats['products_seen'] += len(products)

            self.logger.info('Catalog page=%s yielded %s products', page + 1, len(products))
            if len(products) == 0 and page < 10:
                break

            for product in products:
                row = self.extract_product_meta(product, all_likes_counts, page, search_count, get_images)
                if row and self.validate_listing_row(row):
                    if self._title_matches_wrong_words(row.get('Title'), dictionary.wrong_words):
                        stats['products_excluded_wrong_words'] += 1
                        continue
                    if get_images:
                        row = self._attach_local_catalog_images(row, dictionary.folder)
                    data.append(row)
                    stats['products_valid'] += 1
                else:
                    stats['products_dropped'] += 1

            self.logger.info(
                'Simple scrape progress search=%s page=%s valid_total=%s dropped_total=%s',
                dictionary.folder,
                page + 1,
                stats['products_valid'],
                stats['products_dropped'],
            )

        self.logger.info('Simple scrape summary search=%s stats=%s', dictionary.folder, stats)
        return data

    def remove_not_actually_sold_items(self, new_df, old_df, non_really_sold_items_ids_df_path, sold_df_path, output_folder=None):
        print("I'm removing items that are already present in old_df")
        print(f"Len before dropping duplicates = {len(new_df)}")

        non_really_sold_df = self._load_listing_id_cache(non_really_sold_items_ids_df_path)
        sold_df = self._load_listing_id_cache(sold_df_path)

        if len(new_df) == 0:
            print("No new items to process.")
            return [], old_df, []

        link_list_old = list(old_df["Link"].values)
        link_list_new = list(new_df["Link"].values)

        for link in link_list_new:
            if link in link_list_old:
                old_df.loc[old_df["Link"] == link, "SearchDate"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                new_df = new_df.drop(new_df[new_df['Link'] == link].index)

        new_items_count = len(new_df)
        print(f"Len after dropping duplicates = {new_items_count}")

        items_sold_to_check = old_df.loc[~old_df["Link"].isin(link_list_new)].copy()

        if len(items_sold_to_check) > 0:
            print(f"Before removing non really sold {len(items_sold_to_check)}")
            items_sold_to_check = items_sold_to_check[~items_sold_to_check['Dataid'].isin(non_really_sold_df["Dataid"])]
            print(f"After removing non really sold {len(items_sold_to_check)}")

            if not sold_df.empty:
                before_skip_confirmed = len(items_sold_to_check)
                items_sold_to_check = items_sold_to_check[~items_sold_to_check['Dataid'].isin(sold_df["Dataid"])]
                print(
                    f"After removing already confirmed sold {len(items_sold_to_check)} "
                    f"(skipped {before_skip_confirmed - len(items_sold_to_check)})"
                )

        actually_sold_items = []
        queued_count = 0
        if len(items_sold_to_check) > 0 and output_folder:
            from daily_eventual_sales import enqueue_priority_background_checks

            queued_count = enqueue_priority_background_checks(output_folder, items_sold_to_check, source="compare_and_save")
        print(f"Queued sold-status checks: {queued_count}")
        non_really_sold_df = non_really_sold_df.drop_duplicates(subset=["Dataid"], keep='first')
        non_really_sold_df.to_csv(non_really_sold_items_ids_df_path, index=False)
        print(f"Actually sold items count: {len(actually_sold_items)}")

        return new_df, old_df, actually_sold_items

    def remove_and_manage_old_items_in_search(self, old_df, new_df, unsold_df_path):
        print("I'm about to drop items in the old_df to make space for the new ones")

        new_items_count = len(new_df)

        if os.path.exists(unsold_df_path):
            try:
                unsold_df = pd.read_csv(unsold_df_path)
                print(f"Loaded unsold_df with {len(unsold_df)} items")
            except Exception as e:
                print(f"[WARN] Failed reading {unsold_df_path}: {e}. Initializing empty unsold_df.")
                unsold_df = pd.DataFrame(columns=old_df.columns)
        else:
            print(f"{unsold_df_path} does not exist. Initializing empty unsold_df DataFrame.")
            unsold_df = pd.DataFrame(columns=old_df.columns)

        overflow_count = max(0, len(old_df) + new_items_count - MAX_OLD_ITEMS_PER_SEARCH)
        if overflow_count > 0 and not old_df.empty:
            print(f"Need to drop {overflow_count} stale items from old_df before appending new rows")
            ranked_old_df = self._rank_rows_for_old_df_replacement(old_df)
            rows_to_drop = ranked_old_df.head(overflow_count).drop(columns=[
                '_row_order', '_SearchDateTs'
            ], errors='ignore')
            old_df = old_df.drop(index=rows_to_drop.index)
            unsold_df = pd.concat([unsold_df, rows_to_drop], ignore_index=True)
            print(f"Old_df after stale-row drop: {len(old_df)}")

        dedupe_subset = ["Dataid"] if "Dataid" in unsold_df.columns else ["Link"] if "Link" in unsold_df.columns else None
        if dedupe_subset:
            unsold_df = unsold_df.drop_duplicates(subset=dedupe_subset, keep="first")
        print(f"Saving unsold_df with {len(unsold_df)} items")
        unsold_df.to_csv(unsold_df_path, index=False)

        return old_df, unsold_df

    def compare_and_save_df_serial(self, new_df, old_df, unsold_df_path, sold_df_path,  non_really_sold_items_ids_df_path, output_folder):
        print("in compare and save")

        new_df, old_df, actually_sold_items = self.remove_not_actually_sold_items(
            new_df, old_df, non_really_sold_items_ids_df_path, sold_df_path, output_folder=output_folder
        )

        if not actually_sold_items:
            print("No actually sold items found.")
        else:
            new_sold_df = pd.DataFrame(actually_sold_items)

            if "Dataid" in new_sold_df.columns:
                old_df = old_df[~old_df["Dataid"].isin(new_sold_df["Dataid"])].copy()
                dedupe_subset = ["Dataid"]
            else:
                old_df = old_df[~old_df["Link"].isin(new_sold_df["Link"])].copy()
                dedupe_subset = ["Link"]

            if os.path.exists(sold_df_path):
                sold_df = pd.read_csv(sold_df_path)
                sold_df = pd.concat([sold_df, new_sold_df], ignore_index=True)
            else:
                sold_df = new_sold_df
            sold_df = self._dedupe_listing_rows(sold_df)
            sold_df = sold_df.drop_duplicates(subset=dedupe_subset, keep="last")
            self._write_csv_atomic(sold_df, sold_df_path)
            print(f"Actually sold items: {len(actually_sold_items)}")

        old_df, unsold_df = self.remove_and_manage_old_items_in_search(old_df, new_df, unsold_df_path)

        if len(new_df) == 0:
            print("No new items to process.")
        else:
            print(f"New items to add to old_df: {len(new_df)}")
            print(f"Old_df before concat: {len(old_df)}")

            old_df = pd.concat([old_df, new_df], ignore_index=True)
            print(f"Old_df after concat: {len(old_df)}")
            old_df.drop_duplicates(subset=["Dataid"], keep='first', inplace=True)
            print(f"Old_df after dropping duplicates: {len(old_df)}")
            print(f"saving old_df in {output_folder}/old_df.csv")
            old_df.to_csv(f"{output_folder}/old_df.csv", index=False)
            print("Saved old_df.csv")
