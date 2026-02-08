import traceback
import pandas as pd
import searches
from full_scraper import Full_Scraper
from simple_scraper import Simple_scraper
import sys
import filters as f
import os
import time
import send_batch_items_to_telegram as send_batch_items_to_telegram
from dotenv import load_dotenv
import chat_action
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import utils_scraping
import utils
import filter_items
from analysis_pipeline.vinted_pipeline_incremental import process_new_df

columns_seller = ["SellerId", "SellerName", "Location", "ReviewsCount", "Stars"]
COLUMNS = ['Title','Price','Brand','Size','Link','Likes','Dataid',
           'MarketStatus','SearchDate','Upload_date','Images','SearchCount','Page']

load_dotenv("scripts/telegram_scripts/bot_env.env")

bot_token = os.getenv("BOT_TOKEN")
telegram_chat_id = os.getenv("CHAT_ID")

pages_to_scrape = 1

data_folder = "/home/ale/Desktop/Vinted_New_Version/data/"
old_df_path = os.path.join(data_folder, "old_df.csv")
unsold_df_path = os.path.join(data_folder, "unsold_df.csv")
sold_df_path = os.path.join(data_folder, "sold_df.csv")
seller_df_path = os.path.join(data_folder, "sellers_df.csv")

data_folder_simple_scrape = "/home/ale/Desktop/Vinted_New_Version/data/simple_scrape"

pathfile_simple_old_df = os.path.join(data_folder_simple_scrape, "old_df.csv")
unsold_df_simple_path = os.path.join(data_folder_simple_scrape, "unsold_df.csv")
sold_df_simple_path = os.path.join(data_folder_simple_scrape, "sold_df.csv")
seller_df_simple_path = os.path.join(data_folder_simple_scrape, "sellers_df.csv")

def append_csv_atomic(df_to_add: pd.DataFrame, path: str):
    """Append without race/corruption (single-thread use)."""
    if df_to_add.empty:
        return
    if os.path.exists(path):
        prev = pd.read_csv(path)
        out = pd.concat([prev, df_to_add], ignore_index=True)
    else:
        out = df_to_add
    tmp = path + ".tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)
def scrapeSpecificItems_InSequence(programmed_searches):
    """
    scrape and filter items that passed a series of manual filters
    """
    columns = ['Title', 'Price', 'Brand', 'Size', 'Link', 'Likes', 'Dataid',
                'MarketStatus', 'SearchDate', 'Upload_date', 'Images', "SearchCount", "Page"]

    for search_count in range(1, 500):
        for ricerca in programmed_searches:
            output_folder = os.path.join(data_folder_simple_scrape, ricerca['folder'])
            os.makedirs(output_folder, exist_ok=True)
            pathfile_old_df_item = os.path.join(output_folder, "old_df.csv")

            print(f"SEARCH: {ricerca['search']}")
            print("-" * 20)
            print(f"SEARCH COUNT: {search_count}")

            simple_scraper = Simple_scraper()
            scraped_data = simple_scraper.scrape_products_serial(ricerca, search_count, pages_to_scrape, get_images=True)
            
            print("Scraped data first 5 items:")
            print(scraped_data[:5])

            scraped_df = pd.DataFrame(scraped_data, columns=columns)

            print(f"I Scraped {len(scraped_df)} items")

            old_df = pd.read_csv(pathfile_old_df_item) if os.path.exists(pathfile_old_df_item) else pd.DataFrame(columns=columns)

            items_already_stored = []

            for index, row in scraped_df.iterrows():
                if int(row["Dataid"]) in old_df["Dataid"].values:
                    items_already_stored.append(index)

            new_df = scraped_df.drop(items_already_stored).reset_index(drop=True)
            print(f"From the scraped items ({len(scraped_df)}), {len(new_df)} are new")

            old_df = pd.concat([old_df, new_df], ignore_index=True)
            old_df.to_csv(pathfile_old_df_item, index=False)

            if search_count > 0:
                send_batch_items_to_telegram.send_new_items_to_telegram(new_df, bot_token, telegram_chat_id)
            
            time.sleep(5)  # Sleep to avoid hitting the server too fast

        time.sleep(300) #300  # Sleep to avoid hitting the server too fast

# def _process_one_search(ricerca, search_count, pages_to_scrape):
#     """
#     Runs ONE search (one 'ricerca'): scrape, dedup vs old_df.csv, persist, and return summary + new_df.
#     This function is safe to run in parallel as long as each ricerca has its own folder.
#     """
#     output_folder = os.path.join(data_folder_simple_scrape, ricerca.folder)
#     os.makedirs(output_folder, exist_ok=True)
#     pathfile_old_df_item = os.path.join(output_folder, "old_df.csv")

#     print(f"SEARCH: {ricerca.search}")
#     print("-" * 20)
#     print(f"SEARCH COUNT: {search_count}")

#     # Create a fresh scraper per thread (drivers/parsers are rarely thread-safe)
#     simple_scraper = Simple_scraper()

#     scraped_data = simple_scraper.scrape_products_serial(
#         ricerca, search_count, pages_to_scrape, get_images=True
#     )

#     print(f"Scraped data first 5 items of {ricerca.folder}:")
#     print(scraped_data[:5])

#     scraped_df = pd.DataFrame(scraped_data, columns=COLUMNS)
#     print(f"I Scraped {len(scraped_df)} items")

#     # Load previous items for this ricerca
#     if os.path.exists(pathfile_old_df_item):
#         old_df = pd.read_csv(pathfile_old_df_item)
#     else:
#         old_df = pd.DataFrame(columns=COLUMNS)

    
#     items_already_stored = []

#     for index, row in scraped_df.iterrows():
#         if int(row["Dataid"]) in old_df["Dataid"].values:
#             items_already_stored.append(index)

#     new_df = scraped_df.drop(items_already_stored).reset_index(drop=True)
#     print(f"From the scraped items ({len(scraped_df)}), {len(new_df)} are new")

#     if len(new_df) > 0:
#         assigned_df = process_new_df(
#             new_df,
#             db_path="./out/index.sqlite",
#             price_buffer_size=200
#         )

#         # optional: append assigned rows somewhere
#         stream_path = "./out/stream_assigned.csv"
#         if os.path.exists(stream_path):
#             prev = pd.read_csv(stream_path)
#             pd.concat([prev, assigned_df], ignore_index=True).to_csv(stream_path, index=False)
#         else:
#             assigned_df.to_csv(stream_path, index=False)


#     # print("Im going to filter out the items that are not what i was looking for")
#     # utils.download_images_for_item_analysis(new_df, output_folder)

#     # filter_items.filterOutItemsOutOfDescription(ricerca)



#     # Append and write atomically to avoid partial writes if the process is interrupted
#     combined = pd.concat([old_df, new_df], ignore_index=True)
#     tmp_path = pathfile_old_df_item + ".tmp"
#     combined.to_csv(tmp_path, index=False)
#     os.replace(tmp_path, pathfile_old_df_item)

#     # Return data to the caller (main thread can handle Telegram to avoid rate-limit collisions)
#     return {
#         "ricerca": ricerca,
#         "search_count": search_count,
#         "scraped": len(scraped_df),
#         "new": len(new_df),
#         "new_df": new_df
#     }

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# def scrapeSpecificItems_parallel(
#     programmed_searches,
#     pages_to_scrape=1,
#     bot_token=bot_token,
#     telegram_chat_id=telegram_chat_id,
#     search_workers=4,
#     max_search_counts=500,
#     delay_between_jobs=5,   # NEW: delay (seconds) between submitting each job
# ):
#     """
#     Parallelizes across 'ricerca' for each search_count tick.
#     'search_workers' caps parallelism to be polite to the target site.
#     """
#     for search_count in range(1, max_search_counts + 1):
#         summaries = []

#         # Run all ricerche for this count in parallel (staggered submissions)
#         with ThreadPoolExecutor(max_workers=min(search_workers, len(programmed_searches))) as ex:
#             futures = []
#             for i, ricerca in enumerate(programmed_searches):
#                 futures.append(ex.submit(_process_one_search, ricerca, search_count, pages_to_scrape))
#                 # stagger launches to avoid bursty traffic
#                 if i < len(programmed_searches) - 1:
#                     time.sleep(delay_between_jobs)

#             for fut in as_completed(futures):
#                 try:
#                     summaries.append(fut.result())
#                 except Exception as e:
#                     print(f"[WARN] One search failed: {type(e).__name__}: {repr(e)}")
#                     traceback.print_exc()

#             # # Send Telegram notifications *serially* to avoid API rate limits
#             # for s in summaries:
#             #     if s["search_count"] > 0 and not s["new_df"].empty:
#             #         try:
#             #             send_batch_items_to_telegram.send_new_items_to_telegram(
#             #                 s["new_df"], bot_token, telegram_chat_id
#             #             )
#             #         except Exception as e:
#             #             print(f"[WARN] Telegram send failed for {s['ricerca'].get('search')}: {e}")

#         print(f"Completed search_count {search_count}")
        
#         print(f"Waiting before next search_count...")

#         # Gentle pacing between search iterations
#         time.sleep(600)

##### NEW VERSION WITH ALL THE ANALYSIS IN THE MAIN THREAD  08/02/2026 ######################à

def _process_one_search(ricerca, search_count, pages_to_scrape):
    output_folder = os.path.join(data_folder_simple_scrape, ricerca.folder)
    os.makedirs(output_folder, exist_ok=True)
    pathfile_old_df_item = os.path.join(output_folder, "old_df.csv")

    print(f"SEARCH: {ricerca.search}")
    print("-" * 20)
    print(f"SEARCH COUNT: {search_count}")

    simple_scraper = Simple_scraper()
    scraped_data = simple_scraper.scrape_products_serial(
        ricerca, search_count, pages_to_scrape, get_images=True
    )

    scraped_df = pd.DataFrame(scraped_data, columns=COLUMNS)
    print(f"I Scraped {len(scraped_df)} items")

    if os.path.exists(pathfile_old_df_item):
        old_df = pd.read_csv(pathfile_old_df_item)
    else:
        old_df = pd.DataFrame(columns=COLUMNS)

    # faster than iterrows loop (vectorized)
    if not old_df.empty:
        new_df = scraped_df[~scraped_df["Dataid"].astype(int).isin(old_df["Dataid"].astype(int))].copy()
        new_df.reset_index(drop=True, inplace=True)
    else:
        new_df = scraped_df.copy()

    print(f"From the scraped items ({len(scraped_df)}), {len(new_df)} are new")

    # persist old_df.csv atomically (thread-safe per folder)
    combined = pd.concat([old_df, new_df], ignore_index=True)
    tmp_path = pathfile_old_df_item + ".tmp"
    combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, pathfile_old_df_item)

    return {
        "ricerca": ricerca,
        "search_count": search_count,
        "scraped": len(scraped_df),
        "new": len(new_df),
        "new_df": new_df,              # keep it for main thread
        "output_folder": output_folder # optional
    }


def scrapeSpecificItems_parallel(
    programmed_searches,
    pages_to_scrape=1,
    bot_token=bot_token,
    telegram_chat_id=telegram_chat_id,
    search_workers=4,
    max_search_counts=500,
    delay_between_jobs=5,
):
    stream_path = "./out/stream_assigned.csv"
    db_path = "./out/index.sqlite"

    for search_count in range(1, max_search_counts + 1):
        summaries = []

        # --- SCRAPE (parallel) ---
        with ThreadPoolExecutor(max_workers=min(search_workers, len(programmed_searches))) as ex:
            futures = []
            for i, ricerca in enumerate(programmed_searches):
                futures.append(ex.submit(_process_one_search, ricerca, search_count, pages_to_scrape))
                if i < len(programmed_searches) - 1:
                    time.sleep(delay_between_jobs)

            for fut in as_completed(futures):
                try:
                    summaries.append(fut.result())
                except Exception as e:
                    print(f"[WARN] One search failed: {type(e).__name__}: {repr(e)}")
                    traceback.print_exc()

        # --- ANALYZE (serial, safe) ---
        # process only the new items, and update DB + stream file
        for s in summaries:
            new_df = s["new_df"]
            if new_df is None or new_df.empty:
                continue

            try:
                assigned_df = process_new_df(
                    new_df,
                    db_path=db_path,
                    price_buffer_size=200
                )
                append_csv_atomic(assigned_df, stream_path)

                # If you want: filter deals here and send Telegram now (serial)
                # e.g. top deals:
                # deals = assigned_df[(assigned_df["DealScore"] >= 2.0) & (assigned_df["ProductId"] != -1)]
                # if not deals.empty:
                #     send_batch_items_to_telegram.send_new_items_to_telegram(deals, bot_token, telegram_chat_id)

            except Exception as e:
                print(f"[WARN] Analysis failed for {s['ricerca'].search}: {type(e).__name__}: {e}")
                traceback.print_exc()

        print(f"Completed search_count {search_count}")
        print("Waiting before next search_count...")
        time.sleep(600)
# Worker: create & close its own driver

##### ABOVE NEW VERSION WITH ALL THE ANALYSIS IN THE MAIN THREAD  08/02/2026 ######################à


def _check_sold_with_own_driver(row):
    driver = utils_scraping.make_driver()
    try:
        status, upload_date = "On Sale", "Unknown"
        url = row["Link"]
        html = driver.get_page_content(url, timeout=60, sleep=50)
        if html:
            el_status = html.find('div[data-testid="item-status--content"]', first=True)
            el_upload = html.find('div[itemprop="upload_date"]', first=True)
            if el_upload and getattr(el_upload, "text", None):
                upload_date = el_upload.text
            if el_status and getattr(el_status, "text", "") == "Venduto":
                status = "Sold"
        return row.name, status, upload_date   # return index + result
    
    except Exception:
        return row.name, "On Sale", "Unknown"
    finally:
        try:
            driver.quit()
        except Exception:
            pass

def returnNewSoldItemsInCsv_parallel(csv, max_workers=6, delay=0.5):
    df = pd.read_csv(csv)
    new_sold_items = []
    # ensure columns exist
    if "MarketStatus" not in df.columns:
        df["MarketStatus"] = pd.NA
    if "Upload_date" not in df.columns:
        df["Upload_date"] = pd.NA

    to_check_idx = df.index[df["MarketStatus"] != "Sold"]
    print(f"Checking {len(to_check_idx)} items if they are sold")

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx in to_check_idx:
            futures.append(ex.submit(_check_sold_with_own_driver, df.loc[idx]))
            time.sleep(delay)  # wait before launching the next job

        for fut in as_completed(futures):
            idx, status, upload_date = fut.result()
            if status == "Sold":
                df.at[idx, "MarketStatus"] = status
                df.at[idx, "Upload_date"] = upload_date
                new_sold_items.append(df.loc[idx])
                print(f"Item {df.at[idx, 'Dataid']} is {status.lower()}")

    # df.to_csv(csv, index=False)
    return new_sold_items

def scrapeToGetManuallyFilteredItems(programmed_searches):
    """
    scrape and filter items that passed a series of manual filters
    """
    columns = ['Title', 'Price', 'Brand', 'Size', 'Link', 'Likes', 'Dataid',
                'MarketStatus', 'SearchDate', 'Images', "SearchCount", "Page"]

    for search_count in range(1, 500):
        for ricerca in programmed_searches:
            output_folder = os.path.join(data_folder_simple_scrape, ricerca['folder'])
            os.makedirs(output_folder, exist_ok=True)
            pathfile_old_df_item = os.path.join(output_folder, "old_df.csv")

            print(f"SEARCH: {ricerca['search']}")
            print("-" * 20)
            print(f"SEARCH COUNT: {search_count}")

            simple_scraper = Simple_scraper()
            scraped_data = simple_scraper.scrape_products_serial(ricerca, search_count, pages_to_scrape, get_images=True)
            
            print("Scraped data first 5 items:")
            print(scraped_data[:5])

            scraped_df = pd.DataFrame(scraped_data, columns=columns)

            print(f"I Scraped {len(scraped_df)} items")

            old_df = pd.read_csv(pathfile_old_df_item) if os.path.exists(pathfile_old_df_item) else pd.DataFrame(columns=columns)

            items_already_stored = []

            for index, row in scraped_df.iterrows():
                if int(row["Dataid"]) in old_df["Dataid"].values:
                    items_already_stored.append(index)

            new_df = scraped_df.drop(items_already_stored).reset_index(drop=True)
            print(f"From the scraped items ({len(scraped_df)}), {len(new_df)} are new")

            if search_count > 0:
                send_batch_items_to_telegram.send_new_items_to_telegram(new_df, bot_token, telegram_chat_id)
            
            time.sleep(5)  # Sleep to avoid hitting the server too fast

        time.sleep(300) #300  # Sleep to avoid hitting the server too fast


# sold = returnNewSoldItemsInCsv_parallel("/home/ale/Desktop/Vinted_New_Version/data/simple_scrape/jbl_charge_5/old_df.csv")
# print(sold)