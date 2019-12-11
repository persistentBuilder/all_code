import pandas as pd
# import urllib.request
import requests
from PIL import Image
from io import BytesIO

file_name = "data/faceexp-comparison-data-train-public.csv"
# first_few_lines = pd.read_csv(file_name, nrows=20)
f = open(file_name, "r")
lines = f.readlines()
first_few_lines = lines[0:20]
print(first_few_lines[0])

for line in first_few_lines:
    line_components = line.split(",")
    url = line_components[0][1:-1]
    print(url)
    image_name = url.split("/")[-1]
    print(image_name)

    # response = requests.get(url)
    # img = Image.open(BytesIO(response.content))

    with open(image_name, 'wb') as handle:
        response = requests.get(url, stream=True)

        if not response.ok:
            print(response)

        for block in response.iter_content(1024):
            if not block:
                break
            handle.write(block)
