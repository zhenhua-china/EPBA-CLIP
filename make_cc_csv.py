from pathlib import Path
import pandas as pd
from PIL import Image

project_root = Path(__file__).resolve().parent

image_dir = project_root / "datasets" / "cc_images"
txt_dir = project_root / "datasets" / "cc_txt"
out_csv = project_root / "datasets" / "cc.csv"

image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def read_txt_auto(txt_path):
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return txt_path.read_text(encoding=enc).strip()
        except UnicodeDecodeError:
            continue
    return txt_path.read_text(errors="ignore").strip()

rows = []
missing_txt = []
empty_txt = []
bad_images = []

for img_path in sorted(image_dir.iterdir()):
    if img_path.suffix.lower() not in image_exts:
        continue

    stem = img_path.stem
    txt_path = txt_dir / f"{stem}.txt"

    if not txt_path.exists():
        missing_txt.append(stem)
        continue

    caption = read_txt_auto(txt_path)
    caption = " ".join(caption.split())

    if not caption:
        empty_txt.append(stem)
        continue

    try:
        with Image.open(img_path) as img:
            img.convert("RGB")
    except Exception:
        bad_images.append(str(img_path))
        continue

    file_path = img_path.relative_to(project_root / "datasets").as_posix()

    rows.append({
        "caption": caption,
        "file_path": file_path
    })

df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print("cc.csv 生成完成：", out_csv)
print("有效图文对数量：", len(df))
print("缺少 txt 的图片数量：", len(missing_txt))
print("空 txt 数量：", len(empty_txt))
print("坏图数量：", len(bad_images))

print("\n前 5 条：")
print(df.head())
