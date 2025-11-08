import re

# 输入文件路径
de_path = "train.tags.de-en.de"
en_path = "train.tags.de-en.en"  # 实际路径
out_de = "train.clean.de.txt"
out_en = "train.clean.en.txt"

# 正则表达式：匹配 <...> 标签
tag_re = re.compile(r"<[^>]+>")

def clean_file(in_path, out_path):
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            # 跳过 XML 标签行
            if line.strip().startswith("<"):
                continue
            # 去掉行中可能残留的 <seg> 标签
            clean_line = re.sub(tag_re, "", line).strip()
            if clean_line:
                fout.write(clean_line + "\n")

clean_file(de_path, out_de)
clean_file(en_path, out_en)

print("✅ 清洗完成！")
print(f"德语输出: {out_de}")
print(f"英语输出: {out_en}")
