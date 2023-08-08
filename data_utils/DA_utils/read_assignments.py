import numpy as np
import glob
import os
root_folder = "/home/sateesh/swav_tcn_fs_eval"
for folder in glob.glob(root_folder + "/*"):
    num_clusters = 0
    assigned_clusters = 0
   
    with open(folder+"/final_results.txt") as f:
        lines = f.readlines()
    print("Folder: {}, MOF: {}, F1: {}".format(os.path.basename(folder), lines[0].strip(), lines[-1].strip()))