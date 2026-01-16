import os

def clean_knowledge_base():
    # Configuration
    source_folder = 'knowledge_base_full'
    target_folder = 'new_kb'
    
    # string to identify the line we want to remove
    # Using strip() later handles indentation, so we just need the start text
    remove_marker = "**Categoría**:"

    # Counter for statistics
    files_processed = 0

    print(f"Starting process...")
    print(f"Source: {source_folder}")
    print(f"Target: {target_folder}")

    # Check if source exists
    if not os.path.exists(source_folder):
        print(f"Error: The folder '{source_folder}' was not found.")
        return

    # Walk through the directory tree
    for root, dirs, files in os.walk(source_folder):
        # Determine the relative path to maintain folder structure
        rel_path = os.path.relpath(root, source_folder)
        
        # Create the corresponding target directory
        current_target_dir = os.path.join(target_folder, rel_path)
        os.makedirs(current_target_dir, exist_ok=True)

        for file in files:
            if file.endswith('.md'):
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(current_target_dir, file)

                try:
                    # Read the original file
                    with open(source_file_path, 'r', encoding='utf-8') as f_in:
                        lines = f_in.readlines()

                    # Filter the content
                    cleaned_lines = []
                    for line in lines:
                        # We strip whitespace to check the start, 
                        # just in case there are accidental spaces before **
                        if line.strip().startswith(remove_marker):
                            continue # Skip this line
                        cleaned_lines.append(line)

                    # Write to the new file
                    with open(target_file_path, 'w', encoding='utf-8') as f_out:
                        f_out.writelines(cleaned_lines)
                    
                    files_processed += 1

                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    print("-" * 30)
    print(f"Success! Processed {files_processed} files.")
    print(f"Check the '{target_folder}' directory.")

if __name__ == "__main__":
    clean_knowledge_base()