import asyncio
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import re
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from selenium.webdriver.common.action_chains import ActionChains
import searches as search
import csv
import os
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
import Scraper
import utils

# options = webdriver.ChromeOptions()
# options.add_experimental_option("detach", True)
# options.add_experimental_option("excludeSwitches", ["enable-logging"])
# #1
# options.binary_location = "/usr/bin/google-chrome-stable"   #change to your location 
# #2
# PATH = r'/home/ale/Downloads/chromedriver-linux64/chromedriver' #change also to your location
# service = webdriver.chrome.service.Service(PATH)
# driver = webdriver.Chrome(service=service, options=options)

def select_new_without_bill(driver):
    
    #click condition list menu
    condition_button = driver.find_element(By.CSS_SELECTOR, "[data-testid='catalog--status-filter--trigger']") 
    driver.execute_script("arguments[0].click();", condition_button)
    
    #click new without bill checkbox
    condition_checkbox = driver.find_element(By.XPATH, "//input[@aria-labelledby='status_ids-list-item-1' and @type='checkbox']")
    driver.execute_script("arguments[0].scrollIntoView();", condition_checkbox)
    driver.execute_script("arguments[0].click();", condition_checkbox)


def select_white(driver):    
    #click white checkbox
    white_checkbox = driver.find_element(By.XPATH, "//input[@aria-labelledby='color_ids-list-item-12' and @type='checkbox']")
    driver.execute_script("arguments[0].scrollIntoView();", white_checkbox)
    driver.execute_script("arguments[0].click();", white_checkbox)

def select_black(driver):    
    #click white checkbox
    white_checkbox = driver.find_element(By.XPATH, "//input[@aria-labelledby='color_ids-list-item-1' and @type='checkbox']")
    driver.execute_script("arguments[0].scrollIntoView();", white_checkbox)
    driver.execute_script("arguments[0].click();", white_checkbox)

def find_brand_ids(driver, filters):
    click_brand_menu_list(driver)
    brand_ids = []
    brand_to_add = []
    for filter in filters:
        try:
            search_brand(driver, filter)
            parent_element = driver.find_elements(By.XPATH, f"//span[contains(text(), '{filter}')]/ancestor::div[contains(@class, 'web_ui__Cell__content')]")
            for element in parent_element:
                if element.text.split("(")[0] == filter:
                    parent_parent = element.find_element(By.XPATH, "./ancestor::div[contains(@class, 'web_ui__Cell__cell web_ui__Cell__default web_ui__Cell__navigating')]")
                    brand_id = parent_parent.get_attribute("id").split("-")[-1]
                    brand_ids.append(brand_id)

                    new_brand_row = [filter, brand_id]
                    brand_to_add.append(new_brand_row)
        except:
            print("the try failed.")

    #save the new brands
    temp_df = pd.DataFrame(brand_to_add, columns=["Brand", "Brand_id"])

    df_brand_ids_path = '/home/ale/Desktop/Vinted_New_Version/data/brand_ids.csv'
    
    if not os.path.exists(df_brand_ids_path):
        temp_df.to_csv(df_brand_ids_path, index=False)
    else:
        print("New brands added: ", temp_df)
        brand_df = pd.read_csv(df_brand_ids_path)
        new_brand_df = pd.concat([brand_df, temp_df], ignore_index=True)
        new_brand_df.to_csv(df_brand_ids_path, index=False)
    return brand_ids


def find_color_ids(color_list):
    color_ids = []
    for color in color_list:
        color_ids.append(search.colori[color])
    return color_ids




def set_price_from(driver, value):
    price_input = driver.find_element(By.ID, "price_from")

    price_input.clear()

    price_input.send_keys(f"{value}")

def set_price_to(driver, value):
    price_input = driver.find_element(By.ID, "price_to")

    price_input.clear()

    price_input.send_keys(f"{value}")
    price_input.send_keys(Keys.ENTER)

def sort_items(driver, sorting):
    checkbox_filtro = driver
    if sorting == "bassoAlto":
        checkbox_filtro = driver.find_element(By.XPATH, "//input[@data-testid='sort-by-list-price_low_to_high--input']")
    
    driver.execute_script("arguments[0].scrollIntoView();", checkbox_filtro)
    driver.execute_script("arguments[0].click();", checkbox_filtro)



    
########## clicking  menus below ##################
def click_color_list_menu(driver):
    #click color list menu
    color_menu_button = driver.find_element(By.XPATH, "//button[@data-testid='catalog--color-filter--trigger']") 
   
    driver.execute_script("arguments[0].scrollIntoView();", color_menu_button)
    driver.execute_script("arguments[0].click();", color_menu_button)

def click_sort_list_menu(driver):
    #click color list menu
    sort_menu_button = driver.find_element(By.CSS_SELECTOR, "[data-testid='catalog--sort-filter--trigger']") 
    driver.execute_script("arguments[0].scrollIntoView();", sort_menu_button)
    driver.execute_script("arguments[0].click();", sort_menu_button)


def click_brand_menu_list(driver):
    url = "https://www.vinted.it/catalog?currency=EUR"

    utils.safe_get(driver,url)

    try:
        cookie = driver.find_element(By.ID, "onetrust-accept-btn-handler")
        cookie.click()
    except:
        pass

    brand_menu_button = driver.find_element(By.XPATH, "//button[@data-testid='catalog--brand-filter--trigger']")
    driver.execute_script("arguments[0].scrollIntoView(true);", brand_menu_button)
    driver.execute_script("arguments[0].click();", brand_menu_button)

    print("Ho cliccato il brand menu list, dormo")
    time.sleep(3)

    return driver

# def _open_filters_drawer_if_mobile(driver):
#     # On small/mobile layouts, filters live behind a drawer trigger
#     try:
#         trigger = WebDriverWait(driver, 3).until(
#             EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-testid='catalog--filters--trigger']"))
#         )
#         driver.execute_script("arguments[0].click();", trigger)
#     except TimeoutException:
#         pass  # desktop layout: no drawer

# def click_brand_menu_list(driver):
#     locator = (By.XPATH, "//button[@data-testid='catalog--brand-filter--trigger']")
#     tries = 0
#     while tries < 2:
#         try:
#             _open_filters_drawer_if_mobile(driver)
#             btn = WebDriverWait(driver, 30).until(EC.element_to_be_clickable(locator))
#             driver.execute_script("arguments[0].scrollIntoView();", btn)
#             driver.execute_script("arguments[0].click();", btn)
#             return
#         except (TimeoutException, StaleElementReferenceException):
#             tries += 1

def click_price_menu(driver):
    #click color list menu
    print("clickckckckck")

    price_menu_button = driver.find_element(By.XPATH, "//button[@data-testid='catalog--price-filter--trigger']")
    driver.execute_script("arguments[0].scrollIntoView();", price_menu_button)
    driver.execute_script("arguments[0].click();", price_menu_button)


    
def search_brand(driver, brand_name):
    wait = WebDriverWait(driver, 10)

    # Find only the visible input with that id/class
    xpath = "//input[@id='brand_filter_search' and not(@disabled) and not(contains(@style,'display: none'))]"
    el = wait.until(EC.visibility_of_element_located((By.XPATH, xpath)))

    # Scroll to center to avoid sticky headers
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", el)

    # Try to type normally
    try:
        el.clear()
        el.send_keys(brand_name)
    except Exception:
        # Fallback: set value with JS if typing fails
        driver.execute_script("""
            const input = arguments[0];
            const value = arguments[1];
            input.value = value;
            input.dispatchEvent(new Event('input', {bubbles: true}));
            input.dispatchEvent(new Event('change', {bubbles: true}));
        """, el, brand_name)

    print(f"Ho digitato {brand_name} nel brand search")
    time.sleep(2)

    return driver
















# def tick_brands(filters, driver):
#     brand_piles = driver.find_elements(By.XPATH, "//div[contains(@id, 'brand_ids-list-item')]")
#     for brand in brand_piles:
#         for filter in filters:
#             try:
#                 # Locate the parent element containing both the span with the text "Nike" and the checkbox
#                 nike_parent_element = driver.find_element(By.XPATH, f"//span[contains(text(), '{filter}')]/ancestor::div[contains(@class, 'web_ui__Cell__content')]")
#                 print(nike_parent_element.text)
#                 parent_parent = nike_parent_element.find_element(By.XPATH, "./ancestor::div[contains(@class, 'web_ui__Cell__cell web_ui__Cell__default web_ui__Cell__navigating')]")
#                 # brand_id = parent_parent.get_attribute("id").split("-")[-1]

#                 # print(parent_parent.get_attribute('outerHTML'))
#                 # Find the checkbox within the parent element
#                 nike_checkbox = parent_parent.find_element(By.XPATH, ".//input[@type='checkbox']")

#                 if not nike_checkbox.is_selected():
#                     driver.execute_script("arguments[0].scrollIntoView();", nike_checkbox)
#                     driver.execute_script("arguments[0].click();", nike_checkbox)
#                     print("Nike checkbox clicked successfully.")
#                     break
#             except:
#                 print("the try failed.")