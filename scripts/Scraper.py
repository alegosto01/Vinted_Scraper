import requests
from selenium.webdriver.common.by import By
import time
import pandas as pd
import searches as search
import filters as f
import utils
from selenium.webdriver.chrome.options import Options
import os
from requests_html import HTMLSession
from selenium.webdriver import Remote, ChromeOptions
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from datetime import datetime
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

#scraping browser proxy
AUTH = 'brd-customer-hl_c6889560-zone-scraping_browser1:wu62tqar4piy'
SBR_WEBDRIVER = f'https://{AUTH}@zproxy.lum-superproxy.io:9515'

# data center proxy
# SBR_WEBDRIVER = f'http://brd-customer-hl_c6889560-zone-datacenter_proxy1:9rg06kk55uec@brd.superproxy.io:22225'


#0ce4bdf2c54deb5096b627b4fba5ae18289f93c488150a496190eeb7c6aec936 # api token sivede in web_unlocker 
#0ce4bdf2c54deb5096b627b4fba5ae18289f93c488150a496190eeb7c6aec936
#b400c1d0-9386-4d0f-b2a6-84dd19356b0c # api token non si vede più

api_token = os.getenv("API_TOKEN")
proxy = "http://brd-customer-hl_c6889560-zone-web_unlocker1:@brd.superproxy.io:22225"

# Proxy URL with authentication
class Scraper:
    def __init__(self):
        self.driver = self.init_driver()

    def init_driver(self):
        extension_path = "proxy_auth_extension/proxy_auth_extension.zip"

        chrome_options = Options()
        prefs = {
            "profile.managed_default_content_settings.images": 2,
            "profile.managed_default_content_settings.stylesheets": 2,
            "profile.managed_default_content_settings.fonts": 2,
            "profile.managed_default_content_settings.media_stream": 2,
            "profile.default_content_setting_values.notifications": 2,
            "profile.default_content_setting_values.popups": 2,
            "profile.default_content_setting_values.geolocation": 2,
            "profile.managed_default_content_settings.javascript": 1,
        }
        chrome_options.add_experimental_option("prefs", prefs)
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--window-size=375,667")

        for attempt in range(3):
            try:
                print('Connecting to Scraping Browser...')
                sbr_connection = ChromiumRemoteConnection(SBR_WEBDRIVER, 'goog', 'chrome')
                driver = Remote(sbr_connection, options=ChromeOptions())
                driver.set_page_load_timeout(120)
                print("Connected successfully")
                return driver
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(20)
        print("Failed to load the page after multiple attempts.")
        return None


    def get_page_content_residential(self, url, timeout=40, sleep=10):
        """
        Fetch raw HTML using Bright Data residential proxy
        """

        # ================= Bright Data proxy config =================
        host = 'brd.superproxy.io'
        port = 33335

        username = 'brd-customer-hl_c6889560-zone-residential_proxy1'
        password = 'jt215h4idjjn'

        proxy_url = f'http://{username}:{password}@{host}:{port}'

        proxies = {
            'http': proxy_url,
            'https': proxy_url
        }

        # ================= Session + retries =================
        session = requests.Session()

        retries = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[403, 429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # ================= Headers =================
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.vinted.it/",
            "Connection": "keep-alive"
        }


        max_retries = 3

        for attempt in range(max_retries):
            # ================= Request =================
            try:
                response = session.get(
                    url,
                    headers=headers,
                    proxies=proxies,
                    timeout = (15, 400),   # 15s to connect, 400s to read
                    verify=False
                )

                time.sleep(sleep)

                if response.status_code == 200:
                    return response.text

                else:
                    print(f"[!] Failed: {response.status_code}")
                    print(response.text[:200])
                    return None

            except requests.exceptions.RequestException as e:
                print("[!] Request error:", e)
                return None
    def get_page_content(self, url, timeout=100, sleep=10):
        success = False
        attempts = 0

        #=======================       OLD VERSION       ==========================
        session = HTMLSession()
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        payload = {
            "zone": "web_unlocker1",             
            "url": url,    # Target URL
            "format": "html",                 # Raw HTML format
            "method": "GET",                 # Use the GET method
            "country": "IT"                  # Use a US-based proxy
        }
        headers = {
            "Authorization": api_token,  # Replace with your actual API token
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/"  # or another referring URL if needed
        }

        response = session.get(url, json=payload, headers=headers, timeout=40)

        time.sleep(15)
        if response.status_code == 200:
            # Parse the HTML content from the response
            html = response.html
            # Optionally render JavaScript if the content is dynamically loaded
            max_retries = 3
            retry_count = 0

            # while retry_count < max_retries:
            try:
                html.render(timeout=timeout, sleep=sleep)  # Adjust sleep if needed to allow content to load
                
                # break  # Exit the loop if successful
            except:
                retry_count += 1
                print(f"Timeout occurred. Retrying {retry_count}/{max_retries}...")
                # time.sleep(15)  # Optional: Wait before retrying
            # else:
            #     print("Failed to render page after multiple retries.")
            #     # Add fallback or exit logic
            if html:
                success = True    
                return html
            else:
                print("page not loaded")
                return None
        else:
            print("Failed to retrieve the page:", response.status_code)
            print("Response message:", response.text[:20])  # Print the first 500 characters of the response
    
        
    # create the url setting all the filters of the search
    def create_webpage(self, dictionary): 
        input_search = str(dictionary.search).replace(" ","%20")
        input_search = "&search_text=" + input_search
        #set sorting order
        order = "&order=" + dictionary.sort

        #setting price fro and price to
        price_from = "" if dictionary.prezzoDa == " " else "&price_from=" + dictionary.prezzoDa
        price_to = "" if dictionary.prezzoA == " " else "&price_to=" + dictionary.prezzoA

        #set colors list
        color_search = ""
        if dictionary.colore != " ":
            color_list = dictionary.colore.split("-")
            color_ids = f.find_color_ids(color_list)
            for color_id in color_ids:
                color_search = color_search + "&color_ids[]=" + str(color_id)

        #set brand list
        brands_search = ""

        if dictionary.brands != " ":
            brands_list = dictionary.brands.split("-")
            brands_ids = []
            non_saved_brands = []

            brands_df = pd.read_csv("/home/ale/Desktop/Vinted_New_Version/data/brand_ids.csv")

            # brand_dict = dict(zip(brands_df['brand'], brands_df['brand_id']))

            for brand in brands_list:
                #if i don't already have the brand saved is search it
                if brand not in brands_df["Brand"].values:
                    non_saved_brands.append(brand)
                else: #if i have it i just take it
                    brands_ids.append(brands_df.loc[brands_df['Brand'] == brand, 'Brand_id'].iloc[0])
                
            if non_saved_brands:
                        #setting the input search

                print("creo pagine")
                
                self.driver = self.init_driver()
                #get the page 
                self.driver.get(f"https://www.vinted.it/catalog?currency=EUR{input_search}")
                
                try:
                    cookie = self.driver.find_element(By.ID, "onetrust-accept-btn-handler")
                    cookie.click()
                except:
                    pass
                # with open("output.txt", "w") as file:
                #     file.write(self.driver.page_source)

                print("dormo")
                time.sleep(5)
                print("smetto di dormire")

                print("brand non salvati ancora, li cerco", non_saved_brands)

                brands_ids.extend(f.find_brand_ids(self.driver, non_saved_brands))
                # self.driver.quit()

            for brand_id in brands_ids:
                brands_search = brands_search + "&brand_ids[]=" + str(brand_id)

        #set condition of the items
        condition = ""
        condition_list = dictionary.condition.split("-")
        for elem in condition_list:
            if elem != " ":
                condition = condition + "&status_ids[]=" + elem
        # condition = "" if dictionary["condition"] == " " else "&status_ids[]=" + dictionary["condition"]
        
        #set item's category
        category = "" if dictionary.category == " " else "&catalog[]=" + search.categories[dictionary.category]



        webpage = f"https://www.vinted.it/catalog?currency=EUR{order}{input_search}{color_search}{price_from}{price_to}{condition}{brands_search}{category}"    
        # webpage = "https://www.vinted.it/catalog?search_text=adidas%20gazelle%20black%20and%20white&status_ids[]=1&color_ids[]=12&currency=EUR"
        return webpage


    def get_all_product_images_urls(self, url):
        self.driver = self.init_driver()

        utils.safe_get(self.driver,url)

        #click on one image to open the carousel
        image_button = self.driver.find_element(By.XPATH, "//button[@class='item-thumbnail']")
        self.driver.execute_script("arguments[0].scrollIntoView(true);", image_button)
        self.driver.execute_script("arguments[0].click();", image_button)

        time.sleep(2)
        #get all the images
        image_carousel = self.driver.find_element(By.XPATH, "//div[contains(@class, 'image-carousel__image-wrapper')]")
        images_element = image_carousel.find_elements(By.TAG_NAME, "img")
        image_urls = [img.get_attribute("src") for img in images_element]
        print(f"I got images from url {url}")
        return image_urls

    
    # def compare_and_save_df(self, new_df, old_df, input_search):
    #     # Identifying new items
    #     new_items = new_df[~new_df['Link'].isin(old_df['Link'])]
    #     new_items["MarketStatus"] = "New"

    #     # removed_items = old_df[~old_df['Link'].isin(new_df['Link'])]

    #     # mark sold items as sold
    #     old_df.loc[~old_df['Link'].isin(new_df['Link']), 'MarketStatus'] = 'Sold'


    #     # Save new items
    #     if not new_items.empty:
    #         old_df.append(new_df)
    #         old_df.to_csv(f"{input_search}/{input_search}.csv", index=False)
    #         print(f"New Items: {new_df}")

    #         # notif.sendMessage(f"Nuova Ricerca: {input_search}, {len(new_items)} Nuovi Items")


    #         #send message and download main image

    #         # count = 0
    #         # for index, row in enumerate(new_df):
    #         #     #send whatsapp messages
    #         #     # notif.sendMessage(f"Item {count}: {row.iloc[0]} '  ' {row.iloc[4]}")
    #         #     count += 1
    #         #     #download images
    #         #     data_id = row["Dataid"]
    #         #     img_link = row["Image"]
    #         #     if(img_link != ""):
    #         #         utils.ensure_path_exists(f'{input_search}/{input_search} images')
    #         #         utils.download_image(img_link, f'{input_search}/{input_search} images/{data_id}')



    #     # if not removed_items.empty:
    #     #     for row in removed_items:
    #     #     last_row = utils.get_last_non_empty_row_excel(f"{input_search}/removed_items {input_search}.xlsx")
    #     #     with pd.ExcelWriter(f"{input_search}/removed_items {input_search}.xlsx", engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
    #     #         removed_items.to_excel(writer, sheet_name='Sheet1', index=False, header=True, startrow= ++last_row)
    #     # else:
    #     #     print("nessun articolo è stato venduto")

    #     # Save the current state of the data
    #     # new_df.to_csv(file_path, index=False)


        


# import time
# import random
# from requests.adapters import HTTPAdapter
# from requests.packages.urllib3.util.retry import Retry
# from requests_html import HTMLSession
# from tenacity import (
#     retry,
#     wait_exponential_jitter,
#     stop_after_attempt,
#     retry_if_exception_type,
# )

# # --- 1. Helper: rotate User-Agent ---
# USER_AGENTS = [
#     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36",
#     "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
#     # Add more realistic ASCII-only User-Agents here
# ]

# def random_headers(api_token):
#     return {
#         "Authorization": api_token,
#         "Content-Type": "application/json",
#         "User-Agent": random.choice(USER_AGENTS),
#         "Accept-Language": "en-US,en;q=0.9",
#         "Referer": "https://www.google.com/"
#     }

# # --- 2. Tenacity retry decorator: exponential backoff with jitter on transient errors ---
# @retry(
#     retry=retry_if_exception_type((TimeoutError,)),
#     wait=wait_exponential_jitter(initial=1, max=60),  # 1→60 seconds with randomness
#     stop=stop_after_attempt(5)
# )
# def fetch_through_proxy(api_endpoint, payload, headers, session):
#     resp = session.post(api_endpoint, json=payload, headers=headers, timeout=40)

#     # If the proxy service itself returns 429 or 403, raise to trigger backoff
#     if resp.status_code in (429, 403):
#         # If they give a Retry-After header, honor it:
#         retry_after = resp.headers.get("Retry-After")
#         if retry_after:
#             try:
#                 sleep_secs = int(retry_after)
#                 print(f"Proxy told us to wait {sleep_secs}s")
#                 time.sleep(sleep_secs)
#             except ValueError:
#                 # In case Retry-After is malformed
#                 print(f"Malformed Retry-After header: {retry_after.encode('utf-8', errors='replace').decode('utf-8')}")
#         raise TimeoutError(f"Upstream returned {resp.status_code}")

#     return resp.text

# def scrape_with_resilience(api_token, target_url):
#     session = HTMLSession()

#     # --- 3. Mount a Retry-capable adapter that also retries on 429/403 ---
#     retries = Retry(
#         total=0,  # we’ll handle retries ourselves via tenacity
#         status_forcelist=[500, 502, 503, 504],  
#         allowed_methods=["GET", "POST"],
#     )
#     adapter = HTTPAdapter(max_retries=retries)
#     session.mount("http://", adapter)
#     session.mount("https://", adapter)

#     api_endpoint = f"https://{api_token}.com/web_unlocker1"
#     payload = {
#         "zone": "web_unlocker1",
#         "url": target_url,
#         "format": "html",
#         "method": "GET",
#         "country": "IT"
#     }

#     # random small delay to avoid bursts
#     time.sleep(random.uniform(0.5, 2.0))


#     headers = random_headers(api_token)
#     print("after random headers")
#     html = fetch_through_proxy(api_endpoint, payload, headers, session)
#     print("after fetch through proxy")
#     return html



