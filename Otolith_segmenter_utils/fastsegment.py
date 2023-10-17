import pandas as pd
import numpy as np
import skimage.measure
from general_utils import timer
import scipy as sp
import fastremap

# OpenCV made a great point that for binary images,
# the highest number of provisional labels is
# 1 0  for a 4-connected that's 1/2 the size + 1
# 0 1  for black.
# For 3D six-connected data the same ratio holds
# for a 2x2x2 block, where 1/2 the slots are filled
# in the worst case.
# For 8 connected, since 2x2 bocks are always connected,
# at most 1/4 + 1 of the pixels can be labeled. For 26
# connected, 2x2x2 blocks are connected, so at most 1/8 + 1
# sourdce python CC3d


def retype_downsize(array: np.matrix,maxsize: int):
    min_type = np.min_scalar_type(maxsize)
    if array.dtype == min_type:
        return array
    return array.astype(min_type, copy=False)

def get_segments_from_matrix(matrix: np.matrix,threshold: int = 63, filepath: str = None):
    debug = False
    mask = matrix
    if filepath is not None:
        mask = np.load("/home/max/Projects/test/cod2_top.npy")
    with timer("CPU"):
        if debug:
            print(mask.shape)
        mask = mask > threshold
        # print(mask)
        with timer("SciPy"):
            size = np.count_nonzero(mask) + 1  # +1 for background
            min_type = np.min_scalar_type(size)
            # Shrink types
            mask = mask.astype(min_type, copy=False)
            with timer("label"):
                num_labels = sp.ndimage.label(mask, structure=np.ones((3, 3, 3)), output=mask)
                if debug:
                    print(num_labels)

                # shrink types
                min_type = np.min_scalar_type(num_labels)
                retype_downsize(mask, num_labels)  # finishing refactorying to downsize

        with timer("regionprops"):
            table = skimage.measure.regionprops_table(mask, properties=['label', 'area', 'slice', "centroid"])
            table = pd.DataFrame(table)
            filtered_table = table[table['area'] > 1000]
            sorted_table = filtered_table.sort_values(by=['area'], ascending=False)
            labels, slices = sorted_table['label'].values, sorted_table['slice'].values
            centroids = sorted_table.filter(regex="centroid*").values
            num_islands = len(labels)
            del sorted_table
            if debug:
                print(labels, slices, centroids)
            # relabeling
            label_to_index_dict = dict(zip(list(labels), (np.array(range(1, num_islands + 1), dtype=min_type))))
            mask[~np.isin(mask, labels)] = 0
            min_type = np.min_scalar_type(num_islands)
            fastremap.remap(mask, label_to_index_dict, inplace=True)
            return mask, labels, slices
        print("done")
