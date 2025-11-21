from neurovfm.data import CacheManager

data_dir = "/nfs/turbo/umms-tocho-snr/exp/akhilk/torchmr/raw_data/mri"  # the same as data.data_dir
cache = CacheManager(data_dir)
cache.build_cache(num_workers=8)  # or more, depending on your machine