# save as rebin.py and run with: python rebin.py
import pyBigWig
import numpy as np
import os

BIN_SIZE = 128

INPUT_FILES = [
    '/data/home/s4076520/alphagenome/data/GSM6086873_2_TL_H3K4me3_IP.bw',
    '/data/home/s4076520/alphagenome/data/GSM6086874_3_TL_H3K4me3_IP.bw',
    '/data/home/s4076520/alphagenome/data/GSM6086875_2_TL_H3K27ac_IP.bw',
    '/data/home/s4076520/alphagenome/data/GSM6086876_3_TL_H3K27ac_IP.bw',
]

def rebin_bigwig(input_path, bin_size=BIN_SIZE):
    output_path = input_path.replace('.bw', f'_128bp.bw')
    print(f'Rebinning {os.path.basename(input_path)}...')

    bw_in = pyBigWig.open(input_path)
    bw_out = pyBigWig.open(output_path, 'w')
    chroms = bw_in.chroms()
    bw_out.addHeader(list(chroms.items()))

    for chrom, length in chroms.items():
        n_bins = length // bin_size
        if n_bins == 0:
            continue
        end = n_bins * bin_size
        values = bw_in.stats(chrom, 0, end, nBins=n_bins, type='mean')
        values = np.nan_to_num(np.array(values, dtype=np.float32), nan=0.0)
        starts = list(range(0, end, bin_size))
        ends   = list(range(bin_size, end + bin_size, bin_size))
        bw_out.addEntries([chrom] * n_bins, starts, ends=ends, values=values.tolist())

    bw_in.close()
    bw_out.close()
    print(f'  Saved: {output_path}')

for f in INPUT_FILES:
    rebin_bigwig(f)

print('All done.')