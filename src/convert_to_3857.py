import os
import glob
from concurrent.futures import ProcessPoolExecutor
import subprocess

INPUT_DIR  = r"E:\datasets\outdoor-car-detection-satellite\original"
OUTPUT_DIR = r"E:\datasets\outdoor-car-detection-satellite\epsg_3857"

def process_file(filepath):
    filename = os.path.splitext(os.path.basename(filepath))[0] + ".vrt"
    out_path = os.path.join(OUTPUT_DIR, filename)
    
    if os.path.exists(out_path):
        return
    
    # EPSG:3857 on VRT format
    cmd = [
        "gdalwarp",
        "-of", "VRT",
        "-t_srs", "EPSG:3857",
        "-r", "bilinear",
        "-srcnodata", "0",
        "-dstnodata", "0",
        filepath, out_path
    ]
    
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = glob.glob(os.path.join(INPUT_DIR, "*.tif"))
    print(f"{len(files)} files.")
    
    # max_workers = number of parallel process
    with ProcessPoolExecutor(max_workers=8) as executor:
        list(executor.map(process_file, files))

    print("Completed")

