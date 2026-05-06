import nbformat
import re

file_path = 'commodity_data_collector.ipynb'
with open(file_path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        # Uncomment corn and oil
        if 'COMMODITIES =' in cell.source:
            lines = cell.source.split('\n')
            new_lines = []
            for line in lines:
                if line.startswith('    # {') or line.startswith('    #     ') or line.startswith('    # },'):
                    new_lines.append(line.replace('    # ', '    ', 1))
                else:
                    new_lines.append(line)
            cell.source = '\n'.join(new_lines)
        
        # Update ckpt_key
        if 'def ckpt_key(' in cell.source:
            cell.source = cell.source.replace('def ckpt_key(commodity: str, query_idx: int, year: int) -> str:',
                                              'def ckpt_key(commodity: str, query_idx: int, year: int, month: int) -> str:')
            cell.source = cell.source.replace("return f'{commodity}:q{query_idx}:{year}'",
                                              "return f'{commodity}:q{query_idx}:{year}:{month:02d}'")
            
        # Update fetch_serp_combo signature and tbs
        if 'def fetch_serp_combo(' in cell.source:
            cell.source = cell.source.replace('def fetch_serp_combo(commodity: str, query: str, query_idx: int,\n                     year: int, ckpt: dict) -> tuple[list[dict], int]:',
                                              'import calendar\ndef fetch_serp_combo(commodity: str, query: str, query_idx: int,\n                     year: int, month: int, ckpt: dict) -> tuple[list[dict], int]:')
            cell.source = cell.source.replace('key = ckpt_key(commodity, query_idx, year)',
                                              'key = ckpt_key(commodity, query_idx, year, month)')
            cell.source = cell.source.replace("'tbs':     f'cdr:1,cd_min:01/01/{year},cd_max:12/31/{year}',",
                                              "\'tbs\':     f\'cdr:1,cd_min:{month:02d}/01/{year},cd_max:{month:02d}/{calendar.monthrange(year, month)[1]}/{year}\',")
            cell.source = cell.source.replace("f'    [warn] q{query_idx} {year} page {page}: {e}'",
                                              "f'    [warn] q{query_idx} {year}-{month:02d} page {page}: {e}'")

        # Update loop in the main block
        if 'for year in YEARS:' in cell.source:
            cell.source = cell.source.replace("        for year in YEARS:\n            if budget_exhausted:\n                break\n\n            records, req_count = fetch_serp_combo(name, query, q_idx, year, ckpt)",
                                              "        for year in YEARS:\n            for month in range(1, 13):\n                if budget_exhausted:\n                    break\n\n                records, req_count = fetch_serp_combo(name, query, q_idx, year, month, ckpt)")

        if "total_combos = sum(len(c['serp_queries']) * len(YEARS) for c in COMMODITIES)" in cell.source:
            cell.source = cell.source.replace("total_combos = sum(len(c['serp_queries']) * len(YEARS) for c in COMMODITIES)",
                                              "total_combos = sum(len(c['serp_queries']) * len(YEARS) * 12 for c in COMMODITIES)")

with open(file_path, 'w') as f:
    nbformat.write(nb, f)

print("Notebook updated successfully.")
