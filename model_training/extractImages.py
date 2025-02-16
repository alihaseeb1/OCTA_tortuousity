import os
import shutil

def copy_and_rename_images(root_dir, dest_tortuous, dest_non_tortuous):
    # Make sure the destination directories exist
    os.makedirs(dest_tortuous, exist_ok=True)
    os.makedirs(dest_non_tortuous, exist_ok=True)
    
    count_tortuous = 1
    count_non_tortuous = 1
    
    # Walk through the root directory
    for subdir, dirs, files in os.walk(root_dir):
        # Check if 'tortuous' and 'non_tortuous' directories exist in current directory
        if 'result' in subdir:
            basename = subdir[len(root_dir):].split("\\")[1]
            # print(basename)
            tortuous_dir = os.path.join(subdir, 'tortuous')
            non_tortuous_dir = os.path.join(subdir, 'non_tortuous')
            
            # Copy tortuous images and rename them
            if os.path.exists(tortuous_dir):
                for file in os.listdir(tortuous_dir):
                    if file.endswith(('.jpg', '.png', '.jpeg')):  # Assuming image formats
                        src = os.path.join(tortuous_dir, file)
                        dest = os.path.join(dest_tortuous, f"{basename}_tortuous_{count_tortuous}{os.path.splitext(file)[1]}")
                        shutil.copy2(src, dest)
                        count_tortuous += 1
            
            # Copy non_tortuous images and rename them
            if os.path.exists(non_tortuous_dir):
                for file in os.listdir(non_tortuous_dir):
                    if file.endswith(('.jpg', '.png', '.jpeg')):  # Assuming image formats
                        src = os.path.join(non_tortuous_dir, file)
                        dest = os.path.join(dest_non_tortuous, f"{basename}_non_tortuous_{count_non_tortuous}{os.path.splitext(file)[1]}")
                        shutil.copy2(src, dest)
                        count_non_tortuous += 1
    print("Total tortuous: ", count_tortuous)
    print("Total non_tortuous:", count_non_tortuous)
# Set the root directory and the destination folders
root_directory = "C:\\Users\\aliha\\OCTA_tortuousity\\TV_TUH_processed\\train"  # Modify this path
destination_tortuous = "C:\\Users\\aliha\\OCTA_tortuousity\\tortuous"  # Modify this path
destination_non_tortuous = "C:\\Users\\aliha\\OCTA_tortuousity\\non_tortuous"  # Modify this path

copy_and_rename_images(root_directory, destination_tortuous, destination_non_tortuous)
