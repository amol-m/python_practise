import requests
url = "https://api.kanye.rest"

# response = requests.get(url)

# # print(response.json)
# # print(response.status_code)
# # print(response.content)
# ast.literal_eval(response.text)
url ="https://api.sunrise-sunset.org/json"

parameters = {
    "lat" : 18.5204 ,
    "lng" : 73.8567 ,
    "formatted" : 1 ,
  }

response = requests.get(url= url , params=parameters )
print(response.text)
