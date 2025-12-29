from pathlib import Path
import shutil

src = Path(r'C:\Users\hugoh\OneDrive\Bureau\algo_trader_setup\data')
dst = Path('./data')

copied = 0
for f in src.glob('*. parquet'):
    try:
        shutil.copy2(f, dst)
        copied += 1
        print(f'Copied:  {f. name}')
    except Exception as e:
        print(f'Error copying {f.name}: {e}')

print(f'\n✅ Total copied: {copied} files')