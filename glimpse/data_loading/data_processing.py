import pandas as pd
import os

data_glimpse = "data/processed/"
if not os.path.exists(data_glimpse):
    os.makedirs(data_glimpse)

for year in range (2017, 2021):
    dataset = pd.read_csv(f"data/all_reviews_{year}.csv")
    sub_dataset = dataset[['id','review', 'metareview']]
    sub_dataset.rename(columns={"review": "text", "metareview": "gold"}, inplace=True)
    
    sub_dataset.to_csv(f"{data_glimpse}all_reviews_{year}.csv", index=False)
    
    