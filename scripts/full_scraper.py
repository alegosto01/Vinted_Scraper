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


def scrape_worker(args):
                time.sleep(15)
                """Top‐level for pickling: calls scrape_single_product and tacks its data onto the meta."""
                scraper, meta = args
                new_row, new_seller = scraper.scrape_single_product(
                    url=meta["Link"],
                    data_id=meta["Dataid"],
                    get_images=meta["get_images"]
                )

                # merge into full row
                meta.update({
                    "Images":           new_row.get("Images", []),
                    "Interested_count": new_row.get("Interested_count", 0),
                    "View_count":        new_row.get("View_count", 0),
                    "Description":  new_row.get("Description", ""),
                    "Condition":         new_row.get("Condition", ""),
                    # "Dataid":          new_row.get("Dataid", ""),   
                    "SearchDate":        datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    "MarketStatus":      "On Sale",
                    "Upload_date":       new_row.get("Upload_date", ""),
                    "SellerName":        new_seller.get("SellerName", ""),
                    "SellerId":          new_seller.get("SellerId", ""),
                    "Location":         new_seller.get("Location", ""),
                    "ReviewsCount":    new_seller.get("ReviewsCount", 0),
                    "Stars":           new_seller.get("Stars", -1),
                    # "Page":              page+1,
                    # "SearchCount":       search_count,
                })

                seller_data = {
                    "SellerId": meta["SellerId"],
                    "SellerName": meta["SellerName"],
                    "Location": meta["Location"],
                    "ReviewsCount": meta["ReviewsCount"],
                    "Stars": meta["Stars"]
                }
                return meta, seller_data

class Full_Scraper(Scraper):
    def __init__(self):
        super().__init__()
    

    def fetch_page_and_check(self, item, get_images = False, check_venduto = False, get_upload_date = False):
        time.sleep(15)
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
    
    def scrape_products_serial(self, dictionary, search_count, pages_to_scrape, workers, get_images = False):

        data = []
        seller_rows = []
        page = 0
        last_page = False

        #create the page to scrape
        webpage = self.create_webpage(dictionary)

        #loop through all the pages available
        for i in range(pages_to_scrape):

            new_webpage = webpage + "&page=" + str(page+1)
            print(f"I'm searching in {new_webpage}")

            html_content = self.get_page_content(new_webpage, timeout=60, sleep=5)
            time.sleep(5)
            try:
                element = html_content.find('meta[content="Una community, migliaia di brand e tantissimo stile second-hand. Ti va di iniziare? Ecco come funziona."]', first=True)
            except:
                continue

            # time.sleep(5)

            #if the previous page was empty then stop
            if last_page:
                # page -= 1
                print(f"finished at page {page+1}")
                break
            else:
                print(f"im at page {page+1}")

                    #find list of products in the page
            products = html_content.find('.new-item-box__container')

            all_likes_counts = html_content.find('.u-background-white.u-flexbox.u-align-items-center.new-item-box__favourite-icon')

            #if the page has 0 products mean that we can stop scraping
            print(f"Len products = {len(products)}")
            if len(products) == 0 and page < 10:
                break

            # #get all the data from the products
            # for product in products:

            #     image_link = product.find("img", first=True).attrs.get("src")
            #     if image_link is None:
            #         print("Image link is None")

            #     element_with_data = product.find('.new-item-box__overlay', first=True)

            #     #get link, dataid, and components (which contains tile, price, size and brand)
            #     title = utils.remove_illegal_characters(element_with_data.attrs.get("title"))
            #     components = utils.split_data(title)
            #     link = element_with_data.attrs.get("href")

            #     if "referrer=catalog" not in link:
            #         continue

            #     data_id = element_with_data.attrs.get("data-testid", "").split("-")

            #     if len(data_id) == 7:
            #         data_id = data_id[3]
            #     else:
            #         data_id = data_id[1]

            #     likes_count = 0

            #     for element in all_likes_counts:
            #         # Retrieve the "data-testid" attribute
            #         element_data_test_id = element.attrs.get("data-testid", "")
                    
            #         # Check if data_test_id contains the desired data_id
            #         if data_id and data_id in element_data_test_id:
            #             aria_label = element.attrs.get("aria-label","")
            #             if aria_label != "Aggiungi ai preferiti": # if equal means 0 likes
            #                 likes_count = aria_label.split("Aggiunto ai preferiti da ")[1].split(" ")[0] # Adjust based on actual aria-label structure
            #                 break  # Stop once the correct element is found

            #     #sometimes there is a comma at the end of the price, this code removes it
            #     price = components[1][:-1] if components[1].endswith('.') else components[1]
            #     price = price.replace(',', '.')
            #     price = re.sub(r'[^\d.]', '', price)
            #     if price.count('.') > 1:
            #         parts = price.split('.')
            #         price = parts[0] + '.' + ''.join(parts[1:])  # Keeps the first dot only
                

                
            #     new_row, new_seller_row = self.scrape_single_product(url=link, data_id=str(data_id), get_images=get_images)
                
            #     seller_rows.append(new_seller_row)

            #     new_full_row = {
            #         "Title": components[0],
            #         "Price": float(price) if price else 0.0,
            #         "Brand": components[2],
            #         "Size": components[3],
            #         "Link": link,
            #         "Likes": likes_count,
            #         "Dataid": str(data_id),
            #         "MarketStatus": "On Sale",
            #         "SearchDate": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            #         "Page": page+1,
            #         "SearchCount": search_count,
            #         "Images": image_link,
            #         "Interested_count": new_row["Interested_count"],
            #         "View_count": new_row["View_count"],
            #         "Item_description": new_row["Description"],
            #         "Condition": new_row["Condition"],
            #         "Upload_date": new_row["Upload_date"],
            #         "Dataid": new_row["Dataid"],
            #         "SellerId": " ",
            #         "SellerName": new_row["SellerName"]
            #     }

            #     print(f"new_row = {new_full_row}")
            #     # Append the data to the list
            #     data.append(new_full_row)

                
            #     print(f"data appended = {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

            ###### PARALLEL SCRAPING ######


            def extract_product_meta(product, all_likes_counts, get_images):
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

                return {
                    "Title": components[0],
                    "Price": float(price) if price else 0.0,
                    "Brand": components[2],
                    "Size": components[3],
                    "Link": link,
                    "Likes": likes,
                    "Dataid": data_id,
                    "get_images": get_images,
                    "Page": page + 1,
                    "SearchCount": search_count
                }

            

            # — in your scraping loop —
            metas = []
            for product in products[:20]:
                m = extract_product_meta(product, all_likes_counts, get_images)
                if m:
                    metas.append(m)

            # dispatch to process‐pool
            with ThreadPoolExecutor(max_workers=workers) as exe:
                futures = [
                    exe.submit(scrape_worker, (self, meta))
                    for meta in metas
                ]
                for fut in as_completed(futures):
                    try:
                        meta, seller_data = fut.result()
                        data.append(meta)
                        seller_rows.append(seller_data)

                    except Exception as e:
                        print("Error in worker:", e)

            # `data` is now your list of full rows

            print(f"len data = {len(data)}")
            page += 1


        return data, seller_rows


    #scrape the specific web page of an item
    def scrape_single_product(self, url, data_id, get_images = False): #dictionary was a parameter
        
        html_content = self.get_page_content(url)

        if html_content is None:
            return {}, {}

        page_exists = True              ### ???

        #f the page exists then get all the data, else remove the row from the df (for now)
        if page_exists:
            #get reviews count and star rating
            time.sleep(10)
            try:
                reviews_element = html_content.find("div[class='web_ui__Rating__rating web_ui__Rating__small']", first=True).text
                # reviews_element = html_content.find("h4[class='web_ui__Text__text web_ui__Text__caption web_ui__Text__left']").text
                reviews_count = int(reviews_element)
            except Exception as e:
                print(f"Error getting reviews count: {e} PAGE FAILED TO LOAD")
                reviews_count = 0

            # try:
            try:
                stars_element = html_content.find("div[class='web_ui__Rating__rating web_ui__Rating__small']", first=True)
                stars_text = stars_element.attrs.get("aria-label").split(" ")
                if len(stars_text) == 10:
                    stars = stars_element.attrs.get("aria-label").split(" ")[6]
                else:
                    stars = stars_element.attrs.get("aria-label").split(" ")[2]
            except:
                stars = -1
            # stars = self.driver.find_elements(By.XPATH, "//div[@class='web_ui__Rating__star web_ui__Rating__full']")

            try:
                #get location
                location_element = html_content.find("div[class='u-flexbox u-align-items-baseline']", first=True)

                # html_content.find("div.details-list__item-value--redesign[itemprop='location']", first=True)
                location = location_element.text if location_element else "Not found"            
                # location = self.driver.find_element(By.XPATH, "//div[@class='details-list__item-value' and @itemprop='location']").text
            except:
                location = "Unknown"

            try:
                views_count_element = html_content.find('div.details-list__item-value[itemprop="view_count"] span', first=True)

                # views_count_element = html_content.find("span[class='web_ui__Text__text web_ui__Text__subtitle web_ui__Text__left web_ui__Text__bold']", first=True)
                views_count = int(views_count_element.text) if views_count_element else -1  
                # views_count = int(self.driver.find_element(By.XPATH, "//div[@class='details-list__item-value' and @itemprop='view_count']").text)
            except:
                views_count = -1
            
            
            #get interested people
            try:
                interested_count_element = html_content.find('div.details-list__item-value[itemprop="interested"] span', first=True)
                interested_count = int(interested_count_element.text.split(" ")[0]) if interested_count_element else -1  
                # interested_count = int(self.driver.find_element(By.XPATH, "//div[@class='details-list__item-value' and @itemprop='interested']").text.split(" ")[0])
            except:
                interested_count = -1

            try:
                upload_date = html_content.find('div.details-list__item-value[itemprop="upload_date"] span', first=True).text
                # #get upload date
                # upload_date = " ".join(
                #             html_content.find('div.details-list__item-value[itemprop="upload_date"] span', first=True)
                #     # self.driver.find_element(By.XPATH, "//div[@class='details-list__item-value' and @itemprop='upload_date']").text.split()[:-1]
                #     )
            except:
                upload_date = "Unknown"
            
            try:
                #get item description
                item_description = html_content.find("span[class='web_ui__Text__text web_ui__Text__body web_ui__Text__left web_ui__Text__format']", first=True).text
                # item_description = self.driver.find_element(By.XPATH, "//span[@class='web_ui__Text__text web_ui__Text__body web_ui__Text__left web_ui__Text__format']").text
            except:
                item_description = "Unknown"
            try:
                #get seller name
                seller_name = html_content.find(f"span[data-testid*='profile-username']", first=True).text
            except:
                seller_name = "Unknown"
            # except:
            #     print("problem finding something")
            #     return [], []
            
            try:
                item_condition = html_content.find('div.details-list__item-value[itemprop="status"] span', first=True).text

                # item_condition_element = html_content.find("div[data-testid='item-attributes-status']", first=True)
                # item_condition = item_condition_element.find("div[class='details-list__item-value--redesign']", first=True).text
            except:
                item_condition = "Unknown"

            # print(f"condition = {item_condition}")
            new_seller_row = {
                    "SellerId": " ",
                    "SellerName": seller_name,
                    "Location": location,
                    # "ItemCondition": item_condition,
                    # "ItemDescription": item_description,
                    "ReviewsCount": reviews_count,
                    "Stars": stars
            }

            ### get images or not depending on what is set in the bool parameter get_images
            if get_images:
                try:
                    image_urls = self.get_all_product_images(url)
                except:
                    print("ERROR GETTING IMAGES")
                    image_urls = []
            else:
                image_urls = []

            new_row = {
                "Images": image_urls,
                "Interested_count": interested_count,
                "View_count": views_count,
                "Description": item_description,
                "Condition": item_condition,
                "Upload_date": upload_date,
                # "Dataid": data_id,
                "SellerId": " ",
                "SellerName": seller_name
            }

        else:
            new_row = []
            print("The page doesnt exist")
        
        # print(f"new row = {new_row}")
        return new_row, new_seller_row

        # if len(stars) >= 4 and reviews_count > 3:
        #     print("almeno 4 stelle e 3 reviews")
        # else:
        #     print("non abbastanza stelle o reviews") 

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

        items_sold_to_check = items_sold_to_check[:20]


        if len(items_sold_to_check) > 0:
            print(f"Before removing non really sold {len(items_sold_to_check)}")
            items_sold_to_check = items_sold_to_check[~items_sold_to_check['Dataid'].isin(non_really_sold_items_ids)]
            print(f"After removing non really sold {len(items_sold_to_check)}")
            # items_sold_to_check = items_sold_to_check[~items_sold_to_check['Dataid'].isin(list(full_scraped_df["Dataid"].values))]
        
        # Parallel execution
        actually_sold_items = []

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
        

    def compare_and_save_df_serial(self, new_df, old_df, new_seller_df, old_seller_df, unsold_full_scraped_df, sold_full_scraped_df,  non_really_sold_items_ids):
        print("in compare and save")

        # if len(new_df) > 100:
        #     new_df = new_df[:100]

                  
        # returns updated new_df, old_df and actually_sold_items
        new_df, old_df, actually_sold_items = self.remove_not_actually_sold_items(new_df, old_df, non_really_sold_items_ids)


        if not actually_sold_items:
            print("No actually sold items found.")
            # Store the actully sold items
            new_sold_df = pd.DataFrame(actually_sold_items)        

            sold_full_scraped_df = pd.concat([sold_full_scraped_df, new_sold_df], ignore_index=True)
            sold_full_scraped_df.to_csv(f"/home/ale/Desktop/Vinted_New_Version/data/sold_df.csv", index=False)
            print(f"Actually sold items: {len(actually_sold_items)}")

        # Remove and manage old items in the search
        old_df, unsold_full_scraped_df = self.remove_and_manage_old_items_in_search(old_df, new_df, unsold_full_scraped_df)

        unsold_full_scraped_df.to_csv(f"/home/ale/Desktop/Vinted_New_Version/data/unsold_df.csv", index=False)

        if old_seller_df["SellerId"].notna().any():
            max_id = old_seller_df["SellerId"].max()
        else:
            max_id = 0
            print("No previous seller id")

        for index, row in new_df.iterrows():  
            if row["SellerName"] in old_seller_df["SellerName"].values:
                seller_id = old_seller_df.loc[old_seller_df["SellerName"] == row["SellerName"], "SellerId"].values[0]
                new_df.at[index, "SellerId"] = seller_id  # Modify directly in DataFrame
                new_seller_df.drop(new_seller_df[new_seller_df["SellerName"] == row["SellerName"]].index, inplace=True)
            else:
                max_id += 1
                new_df.at[index, "SellerId"] = max_id  # Modify directly in DataFrame
                new_seller_df.loc[new_seller_df["SellerName"] == row["SellerName"], "SellerId"] = max_id


        old_seller_df = pd.concat([old_seller_df, new_seller_df], ignore_index=True)
        old_seller_df.to_csv(f"/home/ale/Desktop/Vinted_New_Version/data/sellers_df.csv", index=False)            

        if len(new_df) == 0:
            print("No new items to process.")
        else:
            print(old_df.index.duplicated().any())  # True if there are duplicates
            print(new_df.index.duplicated().any())  # True if there are duplicates

            old_df = pd.concat([old_df, new_df], ignore_index=True)
            old_df.to_csv(f"/home/ale/Desktop/Vinted_New_Version/data/old_df.csv", index=False)
            
        print("New seller df:")
        print(new_seller_df)


        # Save new items
        # if len(new_df) > 0:
        #     print("concateno il nuovo dataset") 

        #     print(f"len new_df = {len(new_df)}")
        #     old_df = pd.concat([old_df, new_df], ignore_index=True)
        #     old_df = old_df.drop_duplicates(subset=["Link"], keep='first', inplace=False)
        #     old_df.to_csv(f"prova/old_csv.csv", index=False)
        #     print("finito di concatenare")

            # notif.sendMessage(f"Nuova Ricerca: {input_search}, {len(new_items)} Nuovi Items")


            # send message and download main image

            # count = 0
            # for index, row in enumerate(new_df):
            #     #send whatsapp messages
            #     # notif.sendMessage(f"Item {count}: {row.iloc[0]} '  ' {row.iloc[4]}")
            #     count += 1
            #     #download images
            #     data_id = row["Dataid"]
            #     img_link = row["Image"]
            #     if(img_link != ""):
            #         utils.ensure_path_exists(f'{input_search}/{input_search} images')
            #         utils.download_image(img_link, f'{input_search}/{input_search} images/{data_id}')



        # if not removed_items.empty:
        #     for row in removed_items:
        #     last_row = utils.get_last_non_empty_row_excel(f"{input_search}/removed_items {input_search}.xlsx")
        #     with pd.ExcelWriter(f"{input_search}/removed_items {input_search}.xlsx", engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        #         removed_items.to_excel(writer, sheet_name='Sheet1', index=False, header=True, startrow= ++last_row)
        # else:
        #     print("nessun articolo è stato venduto")

        # Save the current state of the data
        # new_df.to_csv(file_path, index=False)

    #fill the database with additional data scraping every item's webpage



    ##fill the database with additional data scraping every item's webpage
    # def complete_df_with_sigle_scrapes(self, dictionary):

    #     csv_path = f"{dictionary['search']}/{dictionary['search']}.csv"

    #     images_root_folder = f"{dictionary['search']}/{dictionary['search']} images"
        
    #     #read csv to modify it
    #     df = pd.read_csv(csv_path)

    #     print(f"len df = {len(df)}")
    #     #initialize a list of new rows to add to the csv
    #     new_rows = []
    #     new_seller_rows = []

    #     seller_csv_path = "Sellers.csv"
    #     seller_df = pd.read_csv(seller_csv_path)
        

    #     #loop through the dataset to detect the row which are missing the info from the single product scrape
    #     for index, row in df.iterrows():  

    #         # print(f"row = {row['Images']}")
    #         #if images is empty means that the row doesn't have the complete info
    #         #also check that the folder is not created already, if it is we can skip it
    #         # if pd.isna(row["Images"]) and not os.path.exists(f"{dictionary['search']}/{dictionary['search']} images/{row['Dataid']}"):
    #         # if len(list(row["Images"])) == 0:
    #             # self.driver = self.init_driver()

    #             #get the extra data
    #             new_row, new_seller_row = self.scrape_single_product(str(row["Link"]), row["Dataid"])

    #             # if new_row:

    #             #     #maybe this can removed
    #             #     df["Images"] = df["Images"].astype("object")

    #             #     #fill the images cell with the list of images just scraped
    #             #     df.at[index, "Images"] = new_row["Images"]

    #             #     # Remove the images from the new_row becauase right above i populated the column "images" in the original df
    #             #     first_key = next(iter(new_row))
    #             #     new_row.pop(first_key)

    #             #     #download and store all the images
    #             #     image_folder_path = utils.download_all_images(df.at[index, "Images"], dictionary, new_row["Dataid"])
                    
    #             #     #check the images to recognize if the item is what we want
    #             #     is_item_right = dataset_cleaner.check_single_item_images(dictionary, image_folder_path)
    #             # else: #if new_row is empty means that the page doeas exist anymore
    #             #     is_item_right = False

    #             #if the item is correct we store it otherwise we drop the whole row in the csv
    #             new_rows.append(new_row)
    #             new_seller_rows.append(new_seller_row)
    #             # if is_item_right:
    #             #     new_rows.append(new_row)
    #             # else:
    #             #     df.drop(index, inplace=True)  # Drop the row in the main DataFrame
    #         # time.sleep(10)
    #     max_id = seller_df["SellerId"].max()

    #     columns_seller = ["SellerId", "SellerName", "Location", "ReviewsCount", "Stars"]
    #     temp_seller_df = pd.DataFrame(new_seller_rows, columns=columns_seller)
    #     temp_seller_df.drop_duplicates(subset=["SellerName"], keep='first', inplace=True)

        
    #     # seller_df.to_csv(seller_csv_path, index=False)

    #     #create a temporary df with the new rows
    #     columns = ["Interested_count", "View_count", "Item_description", "Upload_date", "Dataid", "SellerId", "SellerName"]
    #     complementary_df = pd.DataFrame(new_rows, columns=columns)

    #     for index, row in complementary_df.iterrows():  
    #         if row["SellerName"] in seller_df["SellerName"].values:
    #             seller_id = seller_df.loc[seller_df["SellerName"] == row["SellerName"], "SellerId"].values[0]
    #             complementary_df.at[index, "SellerId"] = seller_id  # Modify directly in DataFrame
    #             temp_seller_df.drop(temp_seller_df[temp_seller_df["SellerName"] == row["SellerName"]].index, inplace=True)
    #         else:
    #             max_id += 1
    #             complementary_df.at[index, "SellerId"] = max_id  # Modify directly in DataFrame
    #             temp_seller_df.loc[temp_seller_df["SellerName"] == row["SellerName"], "SellerId"] = max_id

    #     #maybe this row can be removed
    #     df.reset_index(drop=True, inplace=True)  # This removes the old index

    #     #add the temporary df with the new rows to the original df
    #     new_df = df.set_index('Dataid').combine_first(complementary_df.set_index('Dataid')).reset_index()
    #     new_df.to_csv(f"{dictionary['search']}/{dictionary['search']}.csv", index=False)
        
        
    #     new_seller_df = pd.concat([seller_df, temp_seller_df], ignore_index=True)
    #     new_seller_df.to_csv(seller_csv_path, index=False)

    #     # add all new images and info to items 
    #     # new_df = pd.merge(df, complementary_df, on="Dataid", how="left")

    #     #overwrite the csv with the updated data
