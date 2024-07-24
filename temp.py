import os

root = "/data/shared/cataract-1K/phase_recognition"
cases = os.listdir(f"{root}/annotations")
all_frames = []

c=0
for c in cases:
    if c != "SYNAPSE_METADATA_MANIFEST.tsv":
        f = pd.read_csv(f"{root}/annotations/{c}/{c}_annotations_phases.csv")
        for _, row in f.iterrows():
            c+=1
print(c)
