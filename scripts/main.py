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
from selenium.webdriver.chrome.options import Options
import os
from selenium.webdriver import Remote, ChromeOptions
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from concurrent.futures import ThreadPoolExecutor, as_completed
import scraping_options
from search_loader import load_searches


# SBR_WEBDRIVER = f'http://brd-customer-hl_c6889560-zone-datacenter_proxy1:9rg06kk55uec@brd.superproxy.io:22225'


searchBenny = 0
searchAle = 1

def main():


    searches = load_searches("/home/ale/Desktop/Vinted_New_Version/data/searches.yaml")

    programmed_searches = [
        s for s in searches.values() if s.enabled
    ]

    scraping_options.scrapeSpecificItems_parallel(programmed_searches, pages_to_scrape=10)



    # for search_count in range(1, 10):

    #     print("-" * 20)
    #     print(f"SEARCH COUNT: {search_count}")

    #     full_scraper = Full_Scraper()
    #     scraped_data, seller_scraped_data = full_scraper.scrape_products_serial(searches.programmed_searches[0], search_count, pages_to_scrape, workers, get_images=False)


    #     print("Scraped data:")
    #     print(scraped_data)
    #     print("Seller scraped data:")
    #     print(seller_scraped_data)


    #     columns = ['Title', 'Price', 'Brand', 'Size', 'Link', 'Likes', 'Dataid',
    #                 'MarketStatus', 'SearchDate', 'Images', "SearchCount", "Page",
    #                 "Interested_count", "View_count", "Description", "Condition",
    #                 "Upload_date", "SellerId", "SellerName", "Stars", "Location", "ReviewsCount"]

    #     columns_seller = ["SellerId", "SellerName", "Location", "ReviewsCount", "Stars"]

    #     new_df = pd.DataFrame(scraped_data, columns=columns)





    #     new_seller_df = pd.DataFrame(seller_scraped_data, columns=columns_seller)
    #     new_seller_df.drop_duplicates(subset=["SellerName"], keep='first', inplace=True)


    #     old_seller_df = pd.read_csv(seller_df_path) if os.path.exists(seller_df_path) else pd.DataFrame(columns=columns_seller)
    #     unsold_full_scraped_df = pd.read_csv(unsold_df_path) if os.path.exists(unsold_df_path) else pd.DataFrame(columns=columns)
    #     sold_full_scraped_df = pd.read_csv(sold_df_path) if os.path.exists(sold_df_path) else pd.DataFrame(columns=columns)
    #     #if it doesn't exists means that is the first search ever


    #     if os.path.exists(old_df_path):
    #         print("not first search i call compare and save")
    #         old_df = pd.read_csv(old_df_path)
    #         old_df.drop(columns=["Dataid.1"], inplace=True, errors='ignore')  # Remove index column if it exists
    #         full_scraper.compare_and_save_df_serial(new_df, old_df, new_seller_df, old_seller_df, unsold_full_scraped_df, sold_full_scraped_df, non_really_sold_items_ids)
    #     else:
    #         old_df = new_df.copy()
    #         old_df.reset_index(drop=True, inplace=True)  # This removes the old index
    #         old_df.to_csv(old_df_path, index=False)
    #         # os.chmod("/home/ale/Desktop/Vinted-Web-Scraper/ /", 0o777)  # Set full read/write/execute permissions for all users
    #         # os.system(f"chown -R {getpass.getuser()}:{getpass.getuser()} {'/home/ale/Desktop/Vinted-Web-Scraper/ /'}")

if __name__ == '__main__':
    main()


