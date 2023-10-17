from well import Well
from skimage import color
import colorsys
class Holder:

    def __init__(self, id_list,well_list,segmentationNode):
        self.id_list = id_list
        self.well_list = well_list
        self.num_wells = len(well_list)
        self.well_colors = ([colorsys.hsv_to_rgb(i / self.num_wells,1,1) for i in range(self.num_wells)])
        self.well_names = [f"well-{i}" for i in range(self.num_wells)]
        self.segmentationNode = segmentationNode
        self.segmentation = segmentationNode.GetSegmentation()
        self.num_segments = len(id_list)
        self.wells = [self._make_well_from_list(well,name,color)
                      for well,name,color in zip(well_list,self.well_names,self.well_colors)]
        #create a smooth rainbow list of colors for each well using lab color space
    def __repr__(self):
        return f"holder({self.id_list},{self.well_list})"
    def _make_well_from_list(self,well_idxs,name,color):
        ids_for_well = [self.id_list[idx] for idx in well_idxs]
        return Well(ids_for_well,name,color,self.segmentationNode)
    def get_wells(self):
        return self.wells

