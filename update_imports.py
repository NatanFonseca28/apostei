import os
import re

project_root = r"d:\Apo$tei"

mapping = {
    'clv': 'core',
    'ev_calculator': 'core',
    'staking': 'core',
    'extractor': 'data',
    'persistence': 'data',
    'models': 'data',
    'feature_engineering': 'data',
    'trainer': 'ml',
    'optimizer': 'ml',
    'pregame_scanner': 'ml',
    'feature_selection': 'ml'
}

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        new_content = content
        for module, folder in mapping.items():
            new_content = re.sub(rf'from\s+src\.{module}\b', f'from src.{folder}.{module}', new_content)
            new_content = re.sub(rf'import\s+src\.{module}\b', f'import src.{folder}.{module}', new_content)

        if "runners" in filepath or "run_" in os.path.basename(filepath):
            if "sys.path.insert" not in new_content:
                sys_path_code = "import os\nimport sys\nsys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))\n\n"
                # To avoid duplicate module imports, just prepend it
                new_content = sys_path_code + new_content

        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated {filepath}")
    except Exception as e:
        print(f"Failed to process {filepath}: {e}")

def main():
    for root, dirs, files in os.walk(project_root):
        if '.venv' in root or '.git' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                process_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
