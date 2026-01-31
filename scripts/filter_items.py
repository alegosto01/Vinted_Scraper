import utils
import os
import pandas as pd
import clip
data_folder_simple_scrape = "/home/ale/Desktop/Vinted_New_Version/data/simple_scrape"


def filterOutItemsOutOfDescription(dictionary):
    for image in os.listdir(os.path.join(data_folder_simple_scrape, dictionary["folder"])):
        if image.endswith(".jpg") or image.endswith(".png"):
            if dictionary['search'] != " ":
                prob_true, prob_false = clip.check_item(option1=dictionary['search'], option2=f"not {dictionary['search']}", image=os.path.join(data_folder_simple_scrape, dictionary["folder"], image))
            else:
                prob_true, prob_false = clip.check_item(option1=dictionary['folder'], option2=f"not {dictionary['folder']}", image=os.path.join(data_folder_simple_scrape, dictionary["folder"], image))    

            print(f"Image: {image}, prob_true: {prob_true}, prob_false: {prob_false}")