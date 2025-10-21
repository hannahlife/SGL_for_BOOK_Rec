import pandas as pd
import os

# === 读取数据 ===
df = pd.read_csv("inter_preliminary_with_2class.csv")

# === 去重：每个用户-物品对只保留一条 ===
df = df.drop_duplicates(subset=['user_id', 'class']).reset_index(drop=True)

# === 将用户重新编号（从0开始） ===
user2id = {u: i for i, u in enumerate(df['user_id'].unique())}
df['user_id'] = df['user_id'].map(user2id)

# === 划分训练集和测试集 ===
train_rows, test_rows = [], []

for user, group in df.groupby('user_id'):
    # 每个用户随机选一条作为测试集，其余为训练集
    test_sample = group.sample(n=1, random_state=42)
    train_sample = group.drop(test_sample.index)
    
    train_rows.append(train_sample)
    test_rows.append(test_sample)

# === 合并结果 ===
train_df = pd.concat(train_rows).reset_index(drop=True)[['user_id', 'class']]
test_df = pd.concat(test_rows).reset_index(drop=True)[['user_id', 'class']]

# === 保存为标准 UI 格式（无表头） ===
os.makedirs("dataset/my_book_class", exist_ok=True)
train_df.to_csv("dataset/my_book_2class/my_book_2class.train", index=False, header=False)
test_df.to_csv("dataset/my_book_2class/my_book_2class.test", index=False, header=False)

# === 统计信息 ===
num_users = len(user2id)
unique_users_in_train = train_df['user_id'].nunique()
unique_users_in_test = test_df['user_id'].nunique()
overlap_users = len(set(train_df['user_id']) & set(test_df['user_id']))

print(f"训练集: {len(train_df)} 条, 测试集: {len(test_df)} 条")
print(f"用户总数: {num_users}")
print(f"训练集中用户数: {unique_users_in_train}")
print(f"测试集中用户数: {unique_users_in_test}")
print(f"训练与测试集中都有记录的用户数: {overlap_users}")

# === 保存用户映射文件 ===
pd.Series(user2id).to_csv("dataset/my_book_2class/user2id_map.csv", header=['user_id'])
print("用户映射文件已保存：dataset/my_book_2class/user2id_map.csv")
