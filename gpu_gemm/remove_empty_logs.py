import os

def delete_empty_txt_files(directory="./clean_logs_2"):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # List all .txt files in the directory
    txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]

    # Loop through each .txt file and check if it's empty
    for file in txt_files:
        file_path = os.path.join(directory, file)
        
        # Check if the file is empty
        if os.path.getsize(file_path) == 0:
            print(f"Deleting empty file: {file_path}")
            os.remove(file_path)

if __name__ == "__main__":
    delete_empty_txt_files()