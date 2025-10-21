import pandas as pd

# === 读取数据 ===
item_df = pd.read_csv("item.csv")
inter_df = pd.read_csv("inter_preliminary.csv")

# === 合并一级分类和二级分类 ===
item_df['combined_class'] = item_df['一级分类'].str.strip() + " | " + item_df['二级分类'].str.strip()

# === 为每个 combined_class 编号，从0开始 ===
class_map = {cls: idx for idx, cls in enumerate(item_df['combined_class'].unique())}
item_df['class'] = item_df['combined_class'].map(class_map)

# === 选取需要的字段 ===
item_class_df = item_df[['book_id', 'class']]

# === 将 class 信息合并到 inter 数据中，保留重复借阅记录 ===
inter_with_class = inter_df.merge(item_class_df, on='book_id', how='left')

# === 检查是否存在未匹配的 book_id ===
missing_books = inter_with_class[inter_with_class['class'].isna()]['book_id'].unique()
if len(missing_books) > 0:
    print(f"有 {len(missing_books)} 个 book_id 在 item.csv 中未找到对应分类。")
    print("缺失 book_id", missing_books)
else:
    print("所有 book_id 都成功匹配到分类。")

# === 保存带类别编号的数据文件 ===
output_file = "inter_preliminary_with_2class.csv"
inter_with_class.to_csv(output_file, index=False, encoding="utf-8-sig")

# === 打印统计信息 ===
print("\n=== 分类聚类结果 ===")
print("合并后的分类总数：", len(class_map))
print("类别编号映射示例：")
for k, v in list(class_map.items())[:20]:  # 打印前20条
    print(f"类别编号 {v} -> {k}")


#保存类别到item
item_output_file = "item_with_class.csv"
item_df.to_csv(item_output_file, index=False, encoding="utf-8-sig")
print(f"已生成带分类编号的 item 文件: {item_output_file}")

# === 打印统计信息 ===
print("\n=== 分类聚类结果 ===")
print("合并后的分类总数：", len(class_map))
print("类别编号映射示例：")
for k, v in list(class_map.items())[:10]:  # 打印前20条
    print(f"类别编号 {v} -> {k}")