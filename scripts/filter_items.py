from numpy import NaN
import utils
import os
import pandas as pd
import clip
import math

data_folder_simple_scrape = "/home/ale/Desktop/Vinted_New_Version/data/simple_scrape"

gpt_hey = "REMOVED_OPENAI_KEY"

def removeRowsContainingWrongWords(df_item, wrong_words):
    df_cleaned = df_item.copy()

    for word in wrong_words:
        df_cleaned = df_cleaned[~df_cleaned['Title'].str.contains(word.strip(), case=False, na=False)]

    print(f"Filtered out {len(df_item) - len(df_cleaned)} items containing wrong words in title.")

    return df_cleaned

def deleteImagesOfFilterOutItemsContainingWrongWords(df_item, df_cleaned, dictionary):
    delete_titles = df_item[~df_item['Title'].isin(df_cleaned['Title'])]
    ids_deleted = delete_titles['Dataid'].tolist()
    ids_deleted = [int(x) for x in ids_deleted if not math.isnan(x)]

    for index, row in delete_titles.iterrows():
        print(f"Deleting item with title: {row['Title']}")

    item_folder = os.path.join(data_folder_simple_scrape, dictionary["folder"])
        
    for dataid in ids_deleted:
        delete_img_path = os.path.join(item_folder, f"{dataid}.jpg")
        if os.path.exists(delete_img_path):
            os.remove(delete_img_path)
            print(f"Deleted image file: {delete_img_path}")
        else:
            print(f"Image file not found, could not delete: {delete_img_path}")

def filterOut_ImageDontMatchTitleItem(dictionary):
    for image in os.listdir(os.path.join(data_folder_simple_scrape, dictionary["folder"])):
        if image.endswith(".jpg") or image.endswith(".png"):
            if dictionary['search'] != " ":
                prob_true, prob_false = clip.check_item(option1=dictionary['search'], option2=f"not {dictionary['search']}", image=os.path.join(data_folder_simple_scrape, dictionary["folder"], image))
            else:
                prob_true, prob_false = clip.check_item(option1=dictionary['folder'], option2=f"not {dictionary['folder']}", image=os.path.join(data_folder_simple_scrape, dictionary["folder"], image))    

            print(f"Image: {image}, prob_true: {prob_true}, prob_false: {prob_false}")

def filterOut_WrongWordsInTitle(dictionary, csv_path):
    df_item = pd.read_csv(csv_path)
    wrong_words = dictionary['wrong_words'].split(',')
    df_cleaned = df_item.copy()


    df_cleaned = removeRowsContainingWrongWords(df_item, wrong_words)

    deleteImagesOfFilterOutItemsContainingWrongWords(df_item, df_cleaned, dictionary)

    df_cleaned.reset_index(drop=True, inplace=True)
    df_cleaned.to_csv(csv_path, index=False)

    return df_item, df_cleaned
