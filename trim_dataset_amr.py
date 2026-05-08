import os

def trim_dataset():
    doc_dir = "output_dataset_doc"
    amr_dir = "output_dataset_amr"

    if not os.path.exists(doc_dir):
        print(f"Error: {doc_dir} does not exist.")
        return

    # Iterate through subfolders in output_dataset_doc ending with .en
    subfolders = [f for f in os.listdir(doc_dir) if os.path.isdir(os.path.join(doc_dir, f)) and f.endswith(".en")]

    for subfolder in subfolders:
        en_path = os.path.join(doc_dir, subfolder)
        vi_subfolder = subfolder.replace(".en", ".vi")
        vi_path = os.path.join(doc_dir, vi_subfolder)
        amr_subfolder_path = os.path.join(amr_dir, subfolder)

        # Check if directories exist
        if not os.path.exists(en_path):
            continue

        # Check files in the .en subfolder
        files = [f for f in os.listdir(en_path) if os.path.isfile(os.path.join(en_path, f))]

        for filename in files:
            file_full_path = os.path.join(en_path, filename)
            
            try:
                with open(file_full_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Error reading {file_full_path}: {e}")
                continue
            
            line_count = len(lines)
            word_count = sum(len(line.split()) for line in lines)

            if line_count > 250 or word_count > 5000:
                # Erase files
                # 1. output_dataset_amr/<en_subfolder>/<file_name>
                amr_file = os.path.join(amr_subfolder_path, filename)
                if os.path.exists(amr_file):
                    os.remove(amr_file)
                
                # 2. output_dataset_doc/<vi_subfolder>/<file_name>
                vi_file = os.path.join(vi_path, filename)
                if os.path.exists(vi_file):
                    os.remove(vi_file)
                
                # 3. output_dataset_doc/<en_subfolder>/<file_name>
                if os.path.exists(file_full_path):
                    os.remove(file_full_path)

                print(f"Removed: {filename} from {subfolder} (Lines: {line_count}, Words: {word_count})")

    # Final count
    print("\nTotal files in output_dataset_amr:")
    if os.path.exists(amr_dir):
        for sub in sorted(os.listdir(amr_dir)):
            sub_p = os.path.join(amr_dir, sub)
            if os.path.isdir(sub_p):
                count = len([f for f in os.listdir(sub_p) if os.path.isfile(os.path.join(sub_p, f))])
                print(f"{sub}: {count} files")
    else:
        print(f"{amr_dir} does not exist.")

if __name__ == "__main__":
    trim_dataset()
