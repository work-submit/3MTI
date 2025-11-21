import json
import sys
# ========== CONFIG ==========
json_path = sys.argv[1]   # 替换为你的 JSON 文件路径
# ============================
print(json_path)
# Load json
with open(json_path, "r") as f:
    data = json.load(f)

# Check data type
assert isinstance(data, dict), "The loaded JSON must be a dict."

# Compute average for each key
averages = {}
for key, val_list in data.items():
    if key == 'in_name':
        continue
    if isinstance(val_list, list) and len(val_list) > 0:
        avg = sum(val_list) / len(val_list)
        averages[key] = avg
    else:
        averages[key] = None  # or set to 0 or skip if empty list

# Print result
for key, avg in averages.items():
    if avg is None:
        continue
    print(f"{key}: {avg}")