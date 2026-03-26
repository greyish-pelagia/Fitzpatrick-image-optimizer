import pandas as pd
import os

df = pd.read_csv(
    "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/refs/heads/main/fitzpatrick17k.csv"
)

img_found = 0
img_not_found = 0

for img in df["md5hash"].values:
    if os.path.exists(
        f"/Users/marcinmalek/Documents/Codes/Projects/Wojtek/Fitzpatric-image-optimizer/data/images/{img}.jpg"
    ):
        img_found += 1
    else:
        img_not_found += 1

print(f"Found {img_found} images")
print(f"Not found {img_not_found} images")
