from slack_sdk import WebClient
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


bot_ID = "U063JAXHQCE"
slack_channel = "D063TDMMU84"

slack_token = 'xoxb-436512348631-6120371602422-xJyE8PBnQqACtIhuaNgU9zpI'
client = WebClient(slack_token)


response = client.conversations_history(channel=slack_channel)
print(len(response['messages']))
print(response['messages'][0]['user'])
print(response['messages'][0]['text'])
print(response['messages'][0]['ts'])
print(response['messages'][1]['ts'])