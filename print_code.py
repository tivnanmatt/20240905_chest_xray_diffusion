import os

def print_code_files():
    # Get the list of files in the current directory
    current_dir = os.getcwd()
    files = os.listdir(current_dir)

    # Filter only Python files
    py_files = [file for file in files if file.endswith('.py')]

    if not py_files:
        print("No Python (.py) files found in the current directory.")
        return

    for py_file in py_files:
        print(f"\n{'=' * 40}\nFile: {py_file}\n{'=' * 40}")
        try:
            # Open the file and print its contents
            with open(py_file, 'r') as file:
                content = file.read()
                print(content)
        except Exception as e:
            print(f"Error reading {py_file}: {e}")

if __name__ == "__main__":
    print_code_files()