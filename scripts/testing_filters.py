import searches 
import filter_items
csv_path = "/home/ale/Desktop/Vinted_New_Version/data/simple_scrape/jbl_charge_5/old_df.csv"

df, df_cleaned = filter_items.filterOut_WrongWordsInTitle(searches.jbl_charge_5, csv_path)

