import pandas as pd
from sklearn.model_selection import train_test_split

# === 读取原始数据 ===
df = pd.read_csv("inter_preliminary.csv")

# === 提取 user_id 和 book_id 两列 ===
df = df[['user_id', 'book_id']].dropna().drop_duplicates()

# === 将用户和图书重新编号（从0开始） ===
user2id = {u: i for i, u in enumerate(df['user_id'].unique())}
book2id = {b: i for i, b in enumerate(df['book_id'].unique())}

df['user_id'] = df['user_id'].map(user2id)
df['book_id'] = df['book_id'].map(book2id)

# === 按用户随机划分训练集与测试集 ===
# 每个用户保留一条记录作为测试数据（常见策略）
train_rows, test_rows = [], []

for user, group in df.groupby('user_id'):
    if len(group) > 1:
        test_sample = group.sample(n=1, random_state=42)
        train_sample = group.drop(test_sample.index)
    else:
        test_sample = group
        train_sample = pd.DataFrame(columns=group.columns)
    
    train_rows.append(train_sample)
    test_rows.append(test_sample)

train_df = pd.concat(train_rows)
test_df = pd.concat(test_rows)

# ===保存为与论文一致的格式 ===
train_df.to_csv("dataset/my_book/book-train.txt", index=False, header=False)
test_df.to_csv("dataset/my_book/book-test.txt", index=False, header=False)

print(f"训练集: {len(train_df)} 条, 测试集: {len(test_df)} 条")
