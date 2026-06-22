import json

# 1. Open your messy JSON file
input_filename = 'Cleaned_NExTLP.json'
output_filename = 'Cleaned_NExTLP.json'

with open(input_filename, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 2. Iterate through the dictionary
for key, value in data.items():
    # 3. Check if the value is a string (meaning it's the markdown version)
    if isinstance(value, str):
        # Strip leading/trailing whitespace
        cleaned_string = value.strip()

        # Remove the ```json at the start and ``` at the end
        if cleaned_string.startswith('```json'):
            cleaned_string = cleaned_string[7:]  # Removes '```json'
        if cleaned_string.endswith('```'):
            cleaned_string = cleaned_string[:-3]  # Removes '```'

        # Strip any remaining newlines (\n) or spaces at the edges
        cleaned_string = cleaned_string.strip()

        # 4. Parse the cleaned string back into a proper dictionary
        try:
            parsed_dict = json.loads(cleaned_string)
            data[key] = parsed_dict
        except json.JSONDecodeError as e:
            print(f"Failed to parse the string at key '{key}'. Error: {e}")

# 5. Save the fixed dictionary to a new JSON file
with open(output_filename, 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print(f"Success! Cleaned JSON saved to {output_filename}")