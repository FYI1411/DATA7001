import os
import subprocess

# Specify the directory containing the PNG files
input_directory = "old_prepost_plot"
output_directory = "old_prepost_plot/svg"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through all files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".png"):
        # Full path to the PNG file
        png_file = os.path.join(input_directory, filename)
        
        # Create the SVG file name by replacing the extension
        svg_file = os.path.join(output_directory, os.path.splitext(filename)[0] + ".svg")
        
        # Use Inkscape command to convert PNG to SVG
        subprocess.run(["inkscape", png_file, "--export-type=svg", "--export-filename=" + svg_file])

        print(f"Converted {filename} to SVG")
