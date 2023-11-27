import json

# Load the .ipynb file
with open('Text_Generation_ShLiDaNa.ipynb', 'r') as f:
    data = json.load(f)

# Extract the code from the .ipynb file
code = ''
for cell in data['cells']:
    if cell['cell_type'] == 'code':
        code += ''.join(cell['source']) + '\n'

# Write the code into a new .py file
with open('Text_Generation_ShLiDaNa.py', 'w') as f:
    f.write(code)