import pandas as pd

train_df = pd.read_csv("dataset/my_book_class/my_book_class.train", header=None, names=['user_id','class'])
test_df = pd.read_csv("dataset/my_book_class/my_book_class.test", header=None, names=['user_id','class'])

# 1检查训练集用户数与范围
print("训练集用户数：", train_df['user_id'].nunique())
print("训练集用户ID范围:", train_df['user_id'].min(), "-", train_df['user_id'].max())

# 2检查是否有用户没有交互
users_all = set(range(train_df['user_id'].max()+1))
users_with_data = set(train_df['user_id'].unique())
missing_users = users_all - users_with_data
print("训练集中无交互的用户数量：", len(missing_users))
print("缺失用户：", list(missing_users)[:])

# 检查测试集用户是否都在训练集
test_only_users = set(test_df['user_id'].unique()) - set(train_df['user_id'].unique())
print("仅出现在测试集的用户：", len(test_only_users))
