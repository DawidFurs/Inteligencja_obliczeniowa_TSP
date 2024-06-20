import gzip
import shutil

with gzip.open('att532.tsp.gz', 'rb') as f_in:
    with open('att532.tsp', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)