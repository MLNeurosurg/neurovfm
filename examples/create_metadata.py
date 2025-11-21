from pathlib import Path
from neurovfm.data import DatasetMetadata

data_root = Path("/nfs/turbo/umms-tocho-snr/exp/akhilk/torchmr/raw_data/mri")
raw_dir = data_root / "raw"

# All BraTS studies are MRI, so map every study dir to "mri"
mode_mapping = {d.name: "mri" for d in raw_dir.iterdir() if d.is_dir()}

metadata = DatasetMetadata.from_directory(data_root, mode_mapping)
metadata.save(data_root / "metadata.json")
print("Saved:", data_root / "metadata.json")