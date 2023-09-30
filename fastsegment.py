import pandas as pd
import skimage as ski
import numpy as np
import skimage.measure
import time
from contextlib import contextmanager
import scipy as sp
from segmentrev import write_numpy_to_segment_file

@contextmanager
def timer(name: str):
    print(name + "...")
    start_time = time.process_time()
    yield
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print(f"{name} took {time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed_time))}")
def retype_downsize(array,maxsize):
    min_type = np.min_scalar_type(maxsize)
    if array.dtype == min_type:
        return array
    return array.astype(min_type, copy=False)


with timer("CPU"):
    mask = np.load("/home/max/Projects/test/cod2_top.npy")
    print(mask.shape)
    mask = mask > 63
    print(mask)
    with timer("SciPy"):
        size = np.count_nonzero(mask) + 1  # +1 for background
        min_type = np.min_scalar_type(size)
        with timer("retype"):
            mask = mask.astype(min_type, copy=False)
        with timer("label"):
            num_labels = sp.ndimage.label(mask, structure=np.ones((3, 3, 3)), output=mask)
            print(num_labels)
            with timer("retype"):
                min_type = np.min_scalar_type(num_labels)
                retype_downsize(mask, num_labels) #finishing refactorying to downsize

    with timer("regionprops"):
        table = skimage.measure.regionprops_table(mask, properties=['label', 'area', 'slice', "centroid"])
        table = pd.DataFrame(table)
        filtered_table = table[table['area'] > 1000]
        sorted_table = filtered_table.sort_values(by=['area'], ascending=False)
        labels, slices = sorted_table['label'].values, sorted_table['slice'].values
        centroids = sorted_table.filter(regex="centroid*").values
        num_islands = len(labels)
        del sorted_table
        print(labels, slices, centroids)

    with timer("relabel"):
        label_to_index_dict = dict(zip(list(labels), (np.array(range(1,num_islands+1),dtype=min_type))))
        mask[~np.isin(mask, labels)] = 0
        min_type = np.min_scalar_type(num_islands)
        newimage = np.zeros_like(mask, dtype=min_type)
        for (index,(label, slice)) in enumerate(zip(labels, slices)):
            newimage[slice] += np.multiply(mask[slice] == label,index+1,dtype=min_type)

    with timer("write"):
        segs_tmpfile = write_numpy_to_segment_file(newimage, num_islands)
        # print(segs_tmpfile)
        # print(segs_tmpfile.name)
        print("done")
    print("done")