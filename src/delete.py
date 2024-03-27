import os

input_folder_path = "../input/clean/"
txt_files = os.listdir(input_folder_path)
if ".DS_Store" in txt_files:
    txt_files.remove(".DS_Store")
print(txt_files)

txt_file = txt_files[0]
with open(input_folder_path + txt_file, 'r', encoding='utf-8') as file:
    text = file.read()

print(text)