# 代码修改总结

## 修改目标
实现训练模型后，输出一个推荐文件，包含：
- **第1列**：原始用户ID（如 119, 1044, 762 等）
- **第2列**：推荐的书籍类别ID

## 数据流程说明

### 1. 数据预处理
- **原始数据**：`inter_preliminary_with_2class.csv` 包含原始用户ID
- **预处理脚本**：`data/my_book_2class_data.py` 
  - 将用户ID重新编号：119 → 0, 1044 → 1, 762 → 2, ...
  - 保存映射关系到：`dataset/my_book_2class/user2id_map.csv`

### 2. 用户映射文件格式
```csv
,user_id
119,0
1044,1
762,2
...
1145,599
```
- 第1列（索引列）：**原始用户ID** (119, 1044, 762, ...)
- 第2列（user_id）：**新用户ID** (0, 1, 2, ..., 599)

### 3. 模型训练
- 使用重新编号的用户ID (0-599) 进行训练
- 数据集：600个用户，1712个书籍类别

## 修改内容

### 修改1：SGL.py - 新增推荐生成方法

**文件**：`model/general_recommender/SGL.py`

**位置**：第 349-428 行

**方法名**：`generate_top1_class_for_all_users(user_mapping_file, output_file)`

**功能**：
1. 读取用户映射文件（`user2id_map.csv`）
2. 为所有用户生成推荐（使用新ID 0-599）
3. 将推荐结果映射回原始用户ID
4. 保存为CSV文件

**关键代码逻辑**：
```python
# 1. 加载映射关系
user_map_df = pd.read_csv(user_mapping_file)
original_user_ids = user_map_df.iloc[:, 0].values  # 原始ID (119, 1044, ...)
new_user_ids = user_map_df.iloc[:, 1].values       # 新ID (0, 1, 2, ...)

# 2. 使用新ID进行预测
scores = self.lightgcn.predict(new_user_ids_tensor).cpu().numpy()
top_class_ids = np.argmax(scores, axis=1)

# 3. 输出使用原始ID
recommend_df = pd.DataFrame({
    'user_id': original_user_ids,         # 原始用户ID
    'recommended_class': top_class_ids    # 推荐的书籍类别
})
```

### 修改2：main.py - 调用推荐生成方法

**文件**：`main.py`

**位置**：第 68-75 行

**修改内容**：
```python
# 训练完成后为所有600个用户生成推荐（使用原始用户ID）
user_mapping_file = "dataset/my_book_2class/user2id_map.csv"
output_file = "dataset/my_book_2class/user_recommendations.csv"

recommender.generate_top1_class_for_all_users(
    user_mapping_file=user_mapping_file,
    output_file=output_file
)
```

## 运行方式

```bash
cd /workspace/SGL_for_Book_Rec
python3 main.py
```

## 输出文件

**文件路径**：`dataset/my_book_2class/user_recommendations.csv`

**文件格式**：
```csv
user_id,recommended_class
58,1234
75,567
119,890
...
1446,1456
```

**说明**：
- `user_id`：原始用户ID（如 119, 1044, 762 等）
- `recommended_class`：推荐的书籍类别ID（0-1711之间）
- 共600行，每个用户一行
- 按原始用户ID升序排列

## 预期输出信息

训练完成后会显示：

```
============================================================
开始为所有用户生成推荐...
============================================================

[1/4] 加载用户映射文件: dataset/my_book_2class/user2id_map.csv
   ✓ 加载了 600 个用户的映射关系
   ✓ 原始用户ID范围: 1 - 1447
   ✓ 新用户ID范围: 0 - 599
   ✓ 有效用户数: 600

[2/4] 准备模型进行预测...
   ✓ 计算最终的用户和物品embeddings...

[3/4] 为 600 个用户生成推荐...
   ✓ 推荐生成完成

[4/4] 保存推荐结果...
============================================================
✓ 推荐结果已保存到: dataset/my_book_2class/user_recommendations.csv
✓ 总共生成 600 条推荐记录
============================================================

前10条推荐记录（原始用户ID）：
user_id  recommended_class
      1               1234
      2                567
      4                890
    ...
```

## 关键改进

1. ✅ **使用原始用户ID**：输出文件中的用户ID是原始ID，不是重新编号的ID
2. ✅ **完整的600个用户**：为所有用户生成推荐
3. ✅ **清晰的输出格式**：CSV文件包含列名，易于阅读
4. ✅ **详细的执行信息**：逐步显示执行进度
5. ✅ **数据验证**：检查用户ID范围，确保数据有效性
