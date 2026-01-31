import pandas as pd
import Scraper
import utils
import time
from simple_scraper import Simple_scraper
import os
from clip import check_item

def compare_item_to_the_similar(row):

    images_folder_path = f"/home/ale/Desktop/Vinted_New_Version/items_analysis/{row['Title']}"
    os.makedirs(images_folder_path, exist_ok=True)


    image_path = os.path.join(images_folder_path, "main_image.jpg")
    utils.download_image(row["Images"], image_path)

    title_image_meatch = check_item(f"the image represents {row['Title']}", f"the image doesn't represent {row['Title']}", image_path)
    blurryness_match = check_item(f"the image is blurry", f"the image is not blurry", image_path)

    print(f"Is the title right?: {title_image_meatch}")
    print(f"Is the image blurry?: {blurryness_match}")

    # Check if the title fits the image


    brand_name = str(row["Brand"]).strip().lower()

    if brand_name in brand_df["Brand"].values:
        brand_id = brand_df.loc[brand_df["Brand"] == str(row["Brand"]).strip().lower(), "Brand_id"].values[0]
    else:
        brand_id = utils.find_brand_id(row["Link"])
        
        brand_df.loc[len(brand_df)] = [brand_name, brand_id]
        brand_df.to_csv("/home/ale/Desktop/Vinted_New_Version/data/brand_ids.csv", index=False)

    row_dict = {
        "search": row["Title"],
        "prezzoDa": " ",
        "prezzoA": " ",
        "condition": "1",
        "colore": " ",
        "brands": f"{brand_name}",
        "sort": "newest_first",
        "category": "Donna/Scarpe"
    }

    ###############################

    # GET ALL SIMILAR PRODUCTS

    ###############################


    simple_scraper = Simple_scraper()
    data = simple_scraper.scrape_products_serial(dictionary=row_dict, search_count=0, pages_to_scrape=10, workers=8, get_images=True)

    columns = ['Title', 'Price', 'Brand', 'Size', 'Link', 'Likes', 'Dataid',
                'MarketStatus', 'SearchDate', 'Images', "SearchCount", "Page"]

    item_search_df = pd.DataFrame(data, columns=columns)

    print(item_search_df)


    #################################

    # FILTER OUT ITEMS THAT ARE NOT RELEVANT FOR THE ANALYSIS

    #################################

    ##################################### FILTER OUT PRICES TOO LOW OR TOO HIGH #####################################

    item_in_analysis_price = row["Price"]

    lower_bound = item_in_analysis_price - (item_in_analysis_price * 0.5)
    upper_bound = item_in_analysis_price + (item_in_analysis_price * 1.5)

    print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

    print(f"Number of items before filtering outliers prices: {len(item_search_df)}")

    item_search_df = item_search_df[(item_search_df["Price"] >= lower_bound) & (item_search_df["Price"] <= upper_bound)]

    print(f"Number of items after filtering outliers prices: {len(item_search_df)}")



    ###################################### FILTER OUT ITEMS THAT ARE NOT THE ITEM WE ARE ANALYSING ######################################


    if len(item_search_df) + 1 == len(os.listdir(images_folder_path)):
        print("All images are already downloaded.")
    else:
        utils.download_images_for_item_analysis(item_search_df, images_folder_path)

    to_remove = []

    for index, row in item_search_df.iterrows():
        # probs = check_item(row["Title"], f"not {row['Title']}", os.path.join(images_folder_path, f"{row['Dataid']}.jpg"))
        probs = check_item(f"the image represents {row['Title']}", f"the image doesn't represent {row['Title']}", os.path.join(images_folder_path, f"{row['Dataid']}.jpg"))

        print(f"is {row['Dataid']} the same item? : {probs}")
        if probs[0] < 0.80:
            print(f"Removing item {row['Dataid']} because it is not the same item.")
            to_remove.append(index)

    item_search_df.drop(to_remove, inplace=True)
    item_search_df.reset_index(drop=True, inplace=True) 

    ######################################

    ## DOWNLOAD IMAGES FOR ITEM ANALYSIS ##

    ######################################



    if len(item_search_df) + 1 == len(os.listdir(images_folder_path)):
        print("All images are already downloaded.")
    else:
        utils.download_images_for_item_analysis(item_search_df, images_folder_path)

    for index, row in item_search_df.iterrows():
        # probs = check_item(row["Title"], f"not {row['Title']}", os.path.join(images_folder_path, f"{row['Dataid']}.jpg"))
        probs = check_item("The image is blurry", "The image is not blurry", os.path.join(images_folder_path, f"{row['Dataid']}.jpg"))
        print(f"Probabilities for item {row['Dataid']}: {probs}")


    #################################

    # ITEMS ANALYSIS

    #################################



    price_mean = item_search_df["Price"].mean()
    price_std = item_search_df["Price"].std()
    item_in_analysis_price = row["Price"]


    print(f"Price Mean: {price_mean}, Price Std: {price_std}")
    print(f"Item in analysis price: {item_in_analysis_price}")

    z_score = (item_in_analysis_price - price_mean) / price_std if price_std != 0 else 0
    mean_comparison = item_in_analysis_price / price_mean if price_mean != 0 else 0
    blurryness_match = blurryness_match[0]  # Assuming blurryness_match is a list or array
    title_image_match = title_image_meatch[0]  # Assuming title_image_meatch

    new_row = {
        "Title": row["Title"],
        "ItemPrice": item_in_analysis_price,
        "MeanComparison": mean_comparison,
        "StandardDeviation": price_std,
        "z_score": z_score,
        "TitleImageMatch": title_image_match,
        "BlurrynessMatch": blurryness_match
    }

    analysis_sold_df.loc[len(analysis_sold_df)] = new_row

analysis_sold_df = pd.DataFrame(columns=['Title', 'ItemPrice', 'MeanComparison', 'StandardDeviation', 'z_score', 'TitleImageMatch', 'BlurrynessMatch'])


sold_df = pd.read_csv("/home/ale/Desktop/Vinted_New_Version/data/simple_scrape/sold_df.csv")
brand_df = pd.read_csv("/home/ale/Desktop/Vinted_New_Version/data/brand_ids.csv")

sold_df = sold_df[:2]

for index, row in sold_df.iterrows():
    print(f"Analyzing item {index + 1}/{len(sold_df)}: {row['Title']}")
    compare_item_to_the_similar(row)


