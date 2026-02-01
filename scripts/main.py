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


# SBR_WEBDRIVER = f'http://brd-customer-hl_c6889560-zone-datacenter_proxy1:9rg06kk55uec@brd.superproxy.io:22225'


searchBenny = 0
searchAle = 1

def main():
    # non_really_sold_items_ids = set()

    # if searchBenny:
    #     scraping_options.scrapeSpecificItems_parallel(searches.programmed_searches_benny)
    if searchAle:
        scraping_options.scrapeSpecificItems_parallel(searches.programmed_searches_ale)

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

    # #if it doesn't exists means that is the first search ever
    # if os.path.exists(f"/home/ale/Desktop/Vinted-Web-Scraper/full_scraped_data/search_df.csv"):
    #     print("not first search i call compare and save")
    #     old_df = pd.read_csv("/home/ale/Desktop/Vinted-Web-Scraper/full_scraped_data/search_df.csv")
    #     full_scraper.compare_and_save_df_serial(new_df,old_df, non_really_sold_items_ids)
    # else:
    #     old_df = new_df.copy()
    #     old_df.reset_index(drop=True, inplace=True)  # This removes the old index
    #     old_df.to_csv("/home/ale/Desktop/Vinted-Web-Scraper/full_scraped_data/search_df.csv", index=False)
    #     # os.chmod("/home/ale/Desktop/Vinted-Web-Scraper/ /", 0o777)  # Set full read/write/execute permissions for all users
    #     # os.system(f"chown -R {getpass.getuser()}:{getpass.getuser()} {'/home/ale/Desktop/Vinted-Web-Scraper/ /'}")

    #     print("first search csv created")

##############################

    # #initialize the output.txt file
    # # output_file = open("output.txt", "w")
    # # sys.stdout = output_file

    # print(f"Date : {datetime.today}")

    # # sys.stdout = sys.__stdout__
    # # output_file.close()


    # path = "/home/ale/Desktop/Vinted-Web-Scraper/ / .csv"
    # big_csv_path = "/home/ale/Desktop/Vinted-Web-Scraper/big_csv/big_csv.csv"

    # if os.path.exists(path):
    #     df = pd.read_csv(path)
    #     big_df = pd.read_csv(big_csv_path)

    #     # df.reset_index(drop=True, inplace=True)
    #     # big_df.reset_index(drop=True, inplace=True)

    #     new_df = pd.concat([df, big_df], ignore_index=True)
    #     new_df.to_csv(big_csv_path, index=False)

    #     try:
    #         shutil.rmtree("/home/ale/Desktop/Vinted-Web-Scraper/ /")
    #     except:
    #         pass
    # times = []

    # for i in range(10):
    #     print(f"Round {i}")
    #     for dictionary in search.programmed_searches:
    #         start_time = time.time()  # Start timer


    #         # Redirect sys.stdout to the file
    #         # output_file = open("output.txt", "a")

    #         # sys.stdout = output_file

    #         tracemalloc.start()

    #         current, peak = tracemalloc.get_traced_memory()

    #         print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    #         print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")

    #         quick_items_scrape(dictionary, i, non_really_sold_items_ids)

    #         # sys.stdout = sys.__stdout__

    #         # # Close the file
    #         # output_file.close()

    #         # After restoring, this will print to the console again
    #         # print("This will be printed on the console.")

    #         tracemalloc.stop()
    #         end_time = time.time()  # End timer
    #         elapsed_time = end_time - start_time  # Calculate elapsed time
    #         times.append(elapsed_time)
    #         print(f"Iteration time: {elapsed_time:.2f} seconds")  # Print time taken for this iteration

    #     time.sleep(10)

    #     # time.sleep(3600)
    # print(times)




# def get_sold_items_slow(path):
#     df = pd.read_csv(path)

# def download_all_images(path):

#     root_folder = "/home/ale/Desktop/Vinted-Web-Scraper/quick_sold_items_images/"
#     already_downloaded = [int(name) for name in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, name))]

#     df = pd.read_csv(path)
#     df = df[df["Images"] != "[]"]
#     df = df[~df["Dataid"].isin(already_downloaded)]
#     print(len(df))
#     counter = 0
#     for index, row in df.iterrows():
#         print(f"Index: {index}")
#         folder_path = os.path.join(root_folder,str(row["Dataid"]))
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#         images = ast.literal_eval(row['Images'])
#         # print(f"data id = {type(images)}")
#         for index_2, image_url in enumerate(images):
#             print(f"Image {index_2 + 1}: {image_url}")
#             utils.download_image(image_url, os.path.join(folder_path,str(index_2)))
#         counter += 1
#         time.sleep(5)
#         # if counter == 50:
#         #     break

# def quick_items_scrape(dictionary, i, non_really_sold_items_ids):
#     scraper = Scraper.Scraper()
#     print(f"search = {dictionary}")
#     input_search = dictionary["search"]
#     # product_root_folder = f"{dictionary['search']}"

#     scraped_data = scraper.scrape_products_serial(dictionary, i)
#     columns = ['Title', 'Price', 'Brand', 'Size', 'Link', 'Likes', 'Dataid',
# 'MarketStatus', 'SearchDate', 'Images', "SearchCount", "Page"]
#     new_df = pd.DataFrame(scraped_data, columns=columns)

#     print(new_df)



#     #if it doesn't exists means that is the first search ever
#     if os.path.exists(f"{input_search}/{input_search}.csv"):
#         print("not first search i call compare and save")
#         old_df = pd.read_csv(f"{input_search}/{input_search}.csv")
#         scraper.compare_and_save_df_serial(new_df,old_df,input_search, non_really_sold_items_ids)
#     else:
#         old_df = new_df.copy()
#         old_df.reset_index(drop=True, inplace=True)  # This removes the old index
#         old_df.to_csv(f"{input_search}/{input_search}.csv", index=False)
#         # os.chmod("/home/ale/Desktop/Vinted-Web-Scraper/ /", 0o777)  # Set full read/write/execute permissions for all users
#         # os.system(f"chown -R {getpass.getuser()}:{getpass.getuser()} {'/home/ale/Desktop/Vinted-Web-Scraper/ /'}")

#         print("first search csv created")




# air_force_1 = {"search":"air force 1 bianche",
#             "prezzoDa":"35",
#             "prezzoA":"70",
#             "status":"1",
#             "colore":"bianco",
#             "brands":"Air Force-Nike Air-Nike",
#             "sort":"newest_first",
#             "category": "scarpe uomo"}

# scraper = Scraper.Scraper(search.air_force_1)


# conv.log_in(scraper.driver)


# Convert the list of dictionaries to a DataFrame




#stop from quitting the page


# def main():
#     # scraper = Scraper.Scraper()


#     proxy_options = {
#         'proxy': {
#             'http': 'http://brd-customer-hl_c6889560-zone-datacenter_proxy1:9rg06kk55uec@brd.superproxy.io:22225',
#             'https': 'https://brd-customer-hl_c6889560-zone-datacenter_proxy1:9rg06kk55uec@brd.superproxy.io:22225',
#             'no_proxy': 'localhost,127.0.0.1'
#         }
#     }

#     # Configure Chrome options
#     chrome_options = webdriver.ChromeOptions()
#     chrome_options.add_argument("--headless")  # Optional: Run in headless mode if needed
#     chrome_options.add_argument("--window-size=1200,800")
#     custom_caps = {
#         'acceptInsecureCerts': True  # Example capability
#     }
#     chrome_options.add_experimental_option("prefs", custom_caps)

#     # Initialize Selenium Wire’s WebDriver with remote WebDriver settings
#     driver = webdriver.Remote(
#         command_executor="http://localhost:4444/wd/hub",  # Connect to the remote Selenium server
#         options=chrome_options,
#         seleniumwire_options=proxy_options  # Pass proxy settings to Selenium Wire
#     )

#     try:
#         # Navigate to the page
#         print("Connecting to Vinted...")
#         driver.get("https://www.vinted.it/catalog?search_text=adidas%20gazelle%20black%20and%20white&status_ids[]=1&color_ids[]=12&currency=EUR&time=1731535560")
#         print("Page loaded")

#         # Wait for the element to be available
#         brand_menu_button = WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.XPATH, "//h1[@class='web_ui__Text__text web_ui__Text__heading web_ui__Text__left']"))
#         )
#         if brand_menu_button:
#             print("Element found:", brand_menu_button)

#     except Exception as e:
#         print("Error occurred:", e)

#     finally:
#         driver.quit()
#         print("Browser closed.")









# for index, row in df_iter.iterrows():
#             if row["Images"] != "" and row["Images"]:
#                 image_urls = row["Images"]
#                 print("good")
#             else:
#                 image_urls = []
#                 print("bad")
#                 break
#             if len(image_urls) > 0:
#             image_folder_path = os.path.join(root_folder,str(row["Dataid"]))
#             if not os.path.exists(image_folder_path):
#                 os.makedirs(image_folder_path)
#             for index_img, image_url in enumerate(image_urls):
#             image_url = image.get_attribute("src")
#                 print(f"Image {index_img + 1}: {image_url}")
#                 print("path exists i wont created it")
#                 utils.download_image(image_url,os.path.join(image_folder_path, str(index_img)))
#             downloaded_dataids.append(int(row["Dataid"]))