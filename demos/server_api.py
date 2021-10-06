#%%
import base64
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

img_path = Path("./samples/paddle_text/resized_sample_image_1_0-0/input.png")

img = Image.open(img_path)
buffered = BytesIO()
img.save(buffered, format="PNG")
img_byte = base64.b64encode(buffered.getvalue())

#%%

# # Detect figure and tabular
rs_data = requests.post(
    "http://localhost:8002/segment/detect-image",
    json={"img_base64": img_byte.decode(), "file_name": f"{img_path.parent.name}.png"},
)

print(rs_data)
print(rs_data.text)
print(rs_data.json())


# %%
# # Detect text
rs_data = requests.post(
    "http://localhost:8002/segment/detect-text",
    json={"img_base64": img_byte.decode(), "file_name": f"{img_path.parent.name}.png"},
)

print(rs_data)
print(rs_data.json())
