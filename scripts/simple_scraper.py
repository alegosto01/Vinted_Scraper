from Scraper import Scraper
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import pandas as pd
import utils
from datetime import datetime
import multiprocessing
import tracemalloc
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed



class Simple_scraper(Scraper):
    def __init__(self):
        super().__init__()
    

    def fetch_page_and_check(self, item, get_images = False, check_venduto = True, get_upload_date = False):
        time.sleep(20)
        try:
            # if int(item["Dataid"]) in non_really_sold_items_ids:
            #     return item, False, "AlreadyChecked"
            url = item["Link"]
            html_content = self.get_page_content(url, timeout=60, sleep=50)

            if html_content:
                # get images solo per recuperare i links delle foto
                if get_images:
                    try:
                        images_links_element = html_content.find('div[class="item-photos"]', first= True)
                        images_links = [img.attrs["src"] for img in images_links_element.find("img") if "src" in img.attrs]

                        item["Images"] = images_links
                    except:
                        print("problemaaaa")
                if check_venduto:
                    element = html_content.find('div[data-testid="item-status--content"]', first=True)
                    if element and element.text == "Venduto":
                        return item, True, "Sold"
                if get_upload_date:
                    try:
                        item["Upload_date"] = html_content.find('div.details-list__item-value[itemprop="upload_date"] span', first=True).text
                    except:
                        item["Upload_date"] = "Unknown"
                        print(f"item = {item['Dataid']} PROBLEMA UPLOAD DATE!!")
            # time.sleep(5)
            return item, False, "On Sale"
        except Exception:
            # time.sleep(3)
            return item, False, "On Sale"

        #scrpe the catagol page and get the main info of the items
    
    def extract_product_meta(self,product, all_likes_counts, page, search_count, get_images):
        """Pulls out title, link, data_id, price, likes, etc.  
        DOES NOT call scrape_single_product yet."""
        # image link
        img = product.find("img", first=True).attrs.get("src")
        element_with_data = product.find('.new-item-box__overlay', first=True)
        title = utils.remove_illegal_characters(element_with_data.attrs["title"])
        components = utils.split_data(title)
        link = element_with_data.attrs["href"]
        if "referrer=catalog" not in link:
            return None  # skip

        # extract data_id    
        parts = element_with_data.attrs.get("data-testid", "").split("-")
        data_id = parts[3] if len(parts)==7 else parts[1]

        # likes
        likes = 0
        for el in all_likes_counts:
            tid = el.attrs.get("data-testid","")
            if data_id in tid:
                lbl = el.attrs.get("aria-label","")
                if lbl!="Aggiungi ai preferiti":
                    likes = int(lbl.split("Aggiunto ai preferiti da ")[1].split()[0])
                break

        # normalize price
        price = components[1]
        if price.endswith('.'): 
            price = price[:-1]
        price = re.sub(r'[^\d.,]', "", price).replace(",", ".")
        if price.count('.')>1:
            p = price.split('.')
            price = p[0] + "." + "".join(p[1:])

        try:
            price = float(price) if price else 0.0
        except ValueError:
            print(f"Error converting price to float: {price}")
            price = 0.0

        return {
            "Title": components[0],
            "Price": price,
            "Brand": components[2],
            "Size": components[3],
            "Link": link,
            "Likes": likes,
            "Dataid": data_id,
            "Images": img,
            "Page": page + 1,
            "SearchCount": search_count,
            "MarketStatus": "On Sale",
            "Processed": False
        }


    def scrape_products_serial(self, dictionary, search_count, pages_to_scrape, get_images = True):

        data = []
        page = 0

        #create the page to scrape
        webpage = self.create_webpage(dictionary)

        #loop through all the pages available
        for i in range(pages_to_scrape):
            new_webpage = webpage + "&page=" + str(page+1)
            print(f"I'm searching in {new_webpage}")

            html_content = self.get_page_content(new_webpage)
            time.sleep(7)

            products = html_content.find('.new-item-box__container')
            all_likes_counts = html_content.find('.u-background-white.u-flexbox.u-align-items-center.new-item-box__favourite-icon')

            #if the page has 0 products mean that we can stop scraping
            print(f"Len products = {len(products)}")
            if len(products) == 0 and page < 10:
                break

            ###### PARALLEL SCRAPING ######


            # — in your scraping loop —
            for product in products:
                row = self.extract_product_meta(product, all_likes_counts, page, search_count, get_images)
                if row:
                    data.append(row)
            print(f"len data = {len(data)}")
            page += 1

        return data

    def remove_not_actually_sold_items(self, new_df, old_df, non_really_sold_items_ids):
        print("I'm removing items that are already present in old_df")
        print(f"Len before dropping duplicates = {len(new_df)}")

        if len(new_df) == 0:
            print("No new items to process.")
            return [], old_df, []

        link_list_old = list(old_df["Link"].values)
        link_list_new = list(new_df["Link"].values)

        # Remove items that are already in old_df
        for link in link_list_new:
            if link in link_list_old:
                old_df.loc[old_df["Link"] == link, "SearchDate"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

                # if old_df:
                new_df = new_df.drop(new_df[new_df['Link'] == link].index)

        new_items_count = len(new_df)
        print(f"Len after dropping duplicates = {new_items_count}")


        # Marking the items that are in old_df but not in new_df (in those there are some fake sold and some real sold)
        for link in link_list_old:
            if link not in link_list_new:
                old_df.loc[old_df["Link"] == link, "MarketStatus"] = "Sold"

        
        print("Marked sold items")


        # Removing items that are not really sold
        items_sold_to_check = old_df.loc[old_df["MarketStatus"] == "Sold"]

        # items_sold_to_check = items_sold_to_check[:20]


        if len(items_sold_to_check) > 0:
            print(f"Before removing non really sold {len(items_sold_to_check)}")
            items_sold_to_check = items_sold_to_check[~items_sold_to_check['Dataid'].isin(non_really_sold_items_ids)]
            print(f"After removing non really sold {len(items_sold_to_check)}")
            # items_sold_to_check = items_sold_to_check[~items_sold_to_check['Dataid'].isin(list(full_scraped_df["Dataid"].values))]
        
        # Parallel execution
        actually_sold_items = []

        if len(items_sold_to_check) > 90:
            items_sold_to_check = items_sold_to_check[:90]

        max_workers = min(8, multiprocessing.cpu_count())

        with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Adjust max_workers based on system/network capacity
            futures = [executor.submit(self.fetch_page_and_check, row)
                    for _, row in items_sold_to_check.iterrows()]

            for future in as_completed(futures):
                item, sold, status = future.result()
                if sold:
                    actually_sold_items.append(item)
                    print(f"Item sold for real: {item['Title']} + {item['Link']}")
                else:
                    if status == "On Sale":
                        try:
                            non_really_sold_items_ids.add(int(item["Dataid"]))
                            old_df.loc[old_df["Link"] == item["Link"], "MarketStatus"] = "On Sale"
                        except:
                            print("problems casting dataid with item")
                    print(f"Item not sold: {item['Title']}")

        print("Parallel scraping complete.")


        return new_df, old_df, actually_sold_items

    def remove_and_manage_old_items_in_search(self, old_df, new_df, unsold_full_scraped):

        print("I'm about to drop items in the old_df to make space for the new ones")


        new_items_count = len(new_df)

        # Make space for the new items deleting the most old ones
        min_search = old_df["SearchCount"].min()
        max_pag = old_df["Page"].max()
        if new_items_count:
            while len(old_df) + new_items_count > 900:
                print(f"Before drop: LEN = {len(old_df)}, New Items Count = {new_items_count}")
                
                print(f"Min SearchCount: {min_search}, Max Page: {max_pag}")
                rows_to_drop = old_df[(old_df["SearchCount"] == min_search) & (old_df["Page"] == max_pag)]
                
                if rows_to_drop.empty:
                    print("No rows to drop, breaking to avoid infinite loop.")
                else:        
                    unsold_full_scraped = pd.concat([unsold_full_scraped, rows_to_drop], ignore_index=True)
                    old_df = old_df.drop(rows_to_drop.index)
                    print(f"After drop: LEN = {len(old_df)}")
                
                if max_pag == 1:
                    min_search += 1
                    max_pag = 10
                else:
                    max_pag -= 1            
        
        return old_df, unsold_full_scraped
        

    def compare_and_save_df_serial(self, new_df, old_df, unsold_full_scraped_df, sold_full_scraped_df,  non_really_sold_items_ids, output_folder):
        print("in compare and save")

        # returns updated new_df, old_df and actually_sold_items
        # new_df, old_df, actually_sold_items = self.remove_not_actually_sold_items(new_df, old_df, non_really_sold_items_ids)


        # if not actually_sold_items:
        #     print("No actually sold items found.")
        #     # Store the actully sold items
        # else:
        #     new_sold_df = pd.DataFrame(actually_sold_items)        
        #     sold_full_scraped_df = pd.concat([sold_full_scraped_df, new_sold_df], ignore_index=True)
        #     sold_full_scraped_df.to_csv(f"{output_folder}/sold_df.csv", index=False)
        #     print(f"Actually sold items: {len(actually_sold_items)}")

        # # Remove and manage old items in the search
        # old_df, unsold_full_scraped_df = self.remove_and_manage_old_items_in_search(old_df, new_df, unsold_full_scraped_df)

        # unsold_full_scraped_df.to_csv(f"{output_folder}/unsold_df.csv", index=False)

        if len(new_df) == 0:
            print("No new items to process.")
        else:
            old_df = pd.concat([old_df, new_df], ignore_index=True)
            old_df.drop_duplicates(subset=["Dataid"], keep='first', inplace=True)
            old_df.to_csv(f"{output_folder}/old_df.csv", index=False)
