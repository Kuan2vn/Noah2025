import os

def print_directory_structure(start_path="."):
    """
    Prints the directory structure starting from the given path.

    Args:
        start_path (str, optional): The path to start printing from. Defaults to the current directory (".").
    """

    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        sub_indent = ' ' * 4 * (level + 1)
        file_count = len(files)
        dir_count = len(dirs)

        combined_items = sorted(dirs + files) # Combine and sort files and dirs for consistent output
        total_items = len(combined_items)

        for index, item in enumerate(combined_items):
            prefix = sub_indent
            if index == total_items - 1: # Last item in the directory
                prefix += '└── '
            else:
                prefix += '├── '

            if item in dirs:
                print('{}{}/'.format(prefix, item)) # Print directory name
            else:
                print('{}{}'.format(prefix, item)) # Print file name

if __name__ == "__main__":
    current_directory = "."  # You can change this to any directory you want to explore
    print(f"Directory structure for: {os.path.abspath(current_directory)}\n")
    print_directory_structure(current_directory)