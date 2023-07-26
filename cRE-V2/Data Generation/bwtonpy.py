import pyBigWig
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

directory_inp = "../../../dataset"

# Generate the list of .bw files
bw_path = directory_inp
bw_files = [os.path.join(bw_path, file) for file in os.listdir(bw_path) if file.endswith('.bw')]
print(bw_files)
bed_path = directory_inp + "/10kb_bedfiles/"
bed_files = [os.path.join(bed_path, file) for file in os.listdir(bed_path) if file.endswith('bins.bed')]

# Load the bed file into a DataFrame
# df_bed = pd.read_csv('/dartfs-hpc/rc/lab/W/WangX/Ruoyun/preprocess/hg38/chr1_2000bins.bed', sep='\t', names=['chrom', 'start', 'end'])




# Function to apply to each row (i.e., genomic bin) in the DataFrame
def fetch_values(row, bw):
    region = (row['chrom'], int(row['start']), int(row['end']))
    values = bw.values(*region)
    transformed_value = np.log(np.sum(values) + 1)
    return transformed_value


for bed_file in tqdm(bed_files):
    df_bed = pd.read_csv(bed_file, sep='\t', names=['chrom', 'start', 'end'])
    chrom=os.path.basename(bed_file).split("_")[0]
    # Initialize a new column in df_bed for each .bw file
    for bw_file in bw_files:
        df_bed[bw_file] = np.nan

    # Loop over the .bw files and apply the function to each row in the DataFrame
    for bw_file in bw_files:
        bw = pyBigWig.open(bw_file)
        df_bed[bw_file] = df_bed.apply(fetch_values, axis=1, args=(bw,))
        bw.close()
        min_val = df_bed[bw_file].min()
        max_val = df_bed[bw_file].max()
        df_bed[bw_file] = (df_bed[bw_file] - min_val) / (max_val - min_val)
    np.save(directory_inp + "/attributes/" + f'{chrom}_attributes_10kb.npy', df_bed[bw_files].values)
