# numbers =[1,2,3]
# new_list = [n+1 for n in numbers]
# print(new_list)

# name ="Angela"
# new_list = [l for l in name]
# print(new_list)

# new_list = [n*2 for n in range(1,5)]
# print(new_list)

# import random
# names =['Alex','Berth','Amit','Roger']
# new_dict = { name : random.randint(1,100) for name in names }
#
# passed_students = { key:value for key,value in new_dict.items() if value > 40}
# print(passed_students)

# sentence = "What is airspeed of flying kite"
# split_sent = sentence.split(' ')
# new_dict = { word:len(word) for word in split_sent}
# print(new_dict)

weather = {'day' : ["Monday" ,"Tuesday" ],
          'temp' : [12, 16]
}

# new_dict = { key:(9/5*value) +32 for (key,value) in weather.items()}
# print(new_dict)

import pandas as pd
weather_df = pd.DataFrame(weather)
for (index,row) in weather_df.iterrows():
    print(row.day, row.temp)
