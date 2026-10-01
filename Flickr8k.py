import os
import json
from collections import defaultdict

root = os.path.dirname(os.path.abspath(__file__))

text_dir = os.path.join(root, "datasets", "Flickr8k", "Flickr8k_text")
token_file = os.path.join(text_dir, "Flickr8k.token.txt")
split_file = os.path.join(text_dir, "Flickr_8k.devImages.txt")


image_rel_dir = r"Flickr8k/Flicker8k_Dataset"

out_json = os.path.join(root, "datasets", "en_val.json")

if not os.path.exists(token_file):
    raise FileNotFoundError(f"找不到 caption 文件: {token_file}")
if not os.path.exists(split_file):
    raise FileNotFoundError(f"找不到划分文件: {split_file}")


caption_dict = defaultdict(list)

with open(token_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        img_id, caption = line.split("\t", 1)
        img_name = img_id.split("#")[0]
        caption_dict[img_name].append(caption)


with open(split_file, "r", encoding="utf-8") as f:
    val_images = [line.strip() for line in f if line.strip()]


annotations = []

missing_caption = []
missing_image = []

for img_name in val_images:
    if img_name not in caption_dict:
        missing_caption.append(img_name)
        continue

    image_path = f"{image_rel_dir}/{img_name}"

    real_image_path = os.path.join(root, "datasets", image_path)
    if not os.path.exists(real_image_path):
        missing_image.append(real_image_path)
        continue

    annotations.append({
        "image": image_path,
        "caption": caption_dict[img_name]
    })

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(annotations, f, ensure_ascii=False, indent=2)

print("保存完成:", out_json)
print("样本数:", len(annotations))
print("缺少 caption:", len(missing_caption))
print("缺少图片:", len(missing_image))

if missing_image[:5]:
    print("缺少图片示例:")
    for p in missing_image[:5]:
        print(p)
