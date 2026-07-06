import requests

url = 'https://static.au.edusercontent.com/files/6gO3V430yabVWe3k4eAT3ylC'
response = requests.get(url)
if response.status_code == 200:
    with open('a2_data.json', 'wb') as file:
        file.write(response.content)
    print('文件下载成功')
else:
    print(f'下载失败，状态码: {response.status_code}')