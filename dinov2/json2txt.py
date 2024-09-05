import json

def json_to_text(json_file, text_file):
    # Open the JSON file and load the data
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Open the text file in write mode
    with open(text_file, 'w') as file:
        # Iterate through each key-value pair in the JSON data
        for key, value in data.items():
            # Write the key and value separated by a space
            file.write(f"{key},{value}\n")

# Example usage
json_to_text('D:\Siladittya_JRF\datasets\ILSVRC\ImageSets\CLS-LOC\labels.json', 'D:\Siladittya_JRF\datasets\ILSVRC\Data\CLS-LOC\labels.txt')