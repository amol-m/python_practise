import pandas as pd

#data_raw = pd.read_csv("weather_data.csv",)
# print(sum(data_raw['temp'])/len(data_raw['temp']))
# print( data_raw[data_raw['temp'].idxmax(axis=0):] )
# print(data_raw[data_raw.temp == data_raw.temp.max()])
#print(data_raw.temp.max())

data_raw = pd.read_csv("2018_Central_Park_Squirrel_Census_-_Squirrel_Data.csv")
# print(data_raw.columns)
data_result = print(data_raw['Primary Fur Color'].value_counts() )



data_dict = {
     "colour" : ["Gray","Red","Black"],
      "count" : ["2473","392","103"]
}

data_dict_df = pd.DataFrame(data_dict)
print(data_dict_df)
