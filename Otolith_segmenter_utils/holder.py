import slicer.util

from well import Well
from skimage import color
import colorsys
class Holder:

    def __init__(self, id_list, well_list, extents,segmentationNode,refVolumeNode):
        self.ids = id_list
        # print("id list",id_list)
        self.well_ids = list(map(lambda well: [id_list[idx] for idx in well],well_list))
        print(self.well_ids,"id list")
        self.extents = extents
        self.num_wells = len(well_list)
        self.well_colors = ([colorsys.hsv_to_rgb(i / self.num_wells,1,1) for i in range(self.num_wells)])
        self.well_names = [f"well-{i}" for i in range(self.num_wells)]
        self.segmentationNode = segmentationNode
        self.segmentation = segmentationNode.GetSegmentation()
        self.volumeNode = refVolumeNode
        self.num_segments = len(id_list)
        # this line is pretty overly big
        self.wells = [self._make_well_from_list(well,name,color)
                      for well,name,color in zip(self.well_ids,self.well_names,self.well_colors)]
        # create a smooth rainbow list of colors for each well using lab color space
    def __repr__(self):
        return f"holder({self.ids},{self.wells})"
    def _make_well_from_list(self,well_ids,name,color):
        print("well ids",well_ids)
        return Well(well_ids,{well_id:self.extents[well_id] for well_id in well_ids},name,color,self.segmentationNode)
    def get_wells(self):
        return self.wells
    def export_wells_to_volume_list(self,mask,volume=None):
        if volume is None:
            volume = slicer.util.arrayFromVolume(self.volumeNode)
        volume_list = []
        for well in self.wells:
            volume_list.append(well.export(volume,mask,self.volumeNode))
        return volume_list
    def export_wells_test(self,mask,volume=None):
        if volume is None:
            volume = slicer.util.arrayFromVolume(self.volumeNode)
        return self.wells[0].export(volume,mask,self.volumeNode)