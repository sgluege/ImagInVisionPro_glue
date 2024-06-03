import os
import shutil
from PIL import Image

# Define the paths for the raw and prepared directories
project_dir = './'
raw_dir = '../' + project_dir + 'data/raw'
prepared_dir = '../' + project_dir + 'data/prepared'
# Loop over all files in the raw directory
for root, dirs, files in os.walk(raw_dir):
    print('Processing', root)
    for file in files:
        # Get the relative path of the file
        rel_path = os.path.relpath(root, raw_dir)
        
        # Create the corresponding directory in the prepared directory
        prepared_subdir = os.path.join(prepared_dir, rel_path)
        os.makedirs(prepared_subdir, exist_ok=True)
        
        # Check if the file is a .bmp image
        if file.endswith('.bmp'):
            # Convert the BMP image to PNG and save it in the prepared directory
            bmp_path = os.path.join(root, file)
            png_path = os.path.join(prepared_subdir, file.replace('.bmp', '.png'))
            
            # Check if a corresponding PNG file already exists
            if not os.path.exists(png_path):
                Image.open(bmp_path).save(png_path, 'PNG')
        else:
            # Copy the file to the prepared directory if it doesn't already exist
            file_path = os.path.join(root, file)
            prepared_file_path = os.path.join(prepared_subdir, file)
            if not os.path.exists(prepared_file_path):
                shutil.copy(file_path, prepared_file_path)
print('Conversion and copying completed.')
