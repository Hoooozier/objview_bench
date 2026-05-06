import json

input_file = "annotated_800k.json"
output_file = "cleaned_attribute.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered = [
    item for item in data
    if item.get("score") == 3
    and item.get("is_multi_object") == "false"
    and item.get("is_scene") == "false"
    and item.get("is_figure") == "false"
    and item.get("is_transparent") == "false"
    and item.get("is_single_color") == "false"
]

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=4)

print(f"Filtering complete, a total of {len(filtered)} entries are retained.")
