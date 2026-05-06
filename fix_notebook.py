import nbformat

file_path = 'commodity_data_collector.ipynb'
with open(file_path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'COMMODITIES =' in cell.source:
            lines = cell.source.split('\n')
            new_lines = []
            for line in lines:
                if line.startswith('    #'):
                    new_lines.append(line.replace('    # ', '    ').replace('    #', '    '))
                else:
                    new_lines.append(line)
            cell.source = '\n'.join(new_lines)

with open(file_path, 'w') as f:
    nbformat.write(nb, f)
print("Notebook uncommented successfully.")
