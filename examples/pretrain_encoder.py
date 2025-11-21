import os
import pandas as pd
import random

data_root = "/nfs/turbo/umms-tocho-snr/exp/akhilk/torchmr/raw_data/mri/processed/"

num_studies = len([f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))])
print(f"Number of studies: {num_studies}")

df_list = []
for study in os.listdir(data_root):
    if os.path.isdir(os.path.join(data_root, study)):
        df_list.append({"study": study, "label1": random.randint(0, 1), "label2": random.randint(0, 1)})

df = pd.DataFrame(df_list)
df.to_csv("/nfs/turbo/umms-tocho/code/akhilk/neurovfm/examples/mil_study_labels.csv", index=False)