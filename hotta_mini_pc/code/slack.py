from slack_sdk import WebClient

slack_token = 'xoxb-436512348631-6120371602422-xJyE8PBnQqACtIhuaNgU9zpI'
client = WebClient(slack_token)
client.chat_postMessage(
    channel="D063R09RFM1",
    text="tweet失敗しました"
)

client.files_upload_v2(
    channel="D063TDMMU84",
    title="本文",
    file="C:\\Users\\hotta_mini\\Desktop\\ちいかわcat\\オエー.PNG",
    initial_comment="投稿です！",
)