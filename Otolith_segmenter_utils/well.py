import numpy as np
import slicer
class Well:
    def __init__(self, ids=[],extents=[],name ="unnamed-well", color = (0,0,0),segmentNode = None):
        #idea have well map ids to positions in the list
        #so that we can quickly find the position of a well
        #alternatively we could have each well printed in order and just not worry about the original list
        self.name = name
        self.ids = ids
        self.extents = extents
        self.color = color
        self.segmentNode = segmentNode
        self.segmentation = segmentNode.GetSegmentation()
        self.segments = [] # List of actual segment pointer objects
        self.num_segments = len(ids)
        self._update_colors()
        self._update_segment_names()
    def __repr__(self):
        return f"well({self.name},{self.ids})"
    def _update_colors(self):
        for id in self.ids:
            self.segmentation.GetSegment(id).SetColor(self.color)
    def _update_segment_names(self):
        for idx,id in enumerate(self.ids):
            self.segmentation.GetSegment(id).SetName(f"{self.name}-{idx}")
    def set_well_color(self,color):
        self.color = color
        self._update_colors()
    def set_name(self,old_name,new_name):
        self.name = new_name
        self._update_segment_names()
    def get_ids(self):
        return self.real_ids
    def get_name(self):
        return self.name
    def get_segment(self,idx):
        if idx not in self.id_to_position:
            self.id_to_position[idx] = self.real_ids.index(idx)
        return self.segmentation.GetSegment(self.ids[idx])
    def export(self,volume,labelmap,volumeNode):
        logic = slicer.modules.OtolithSegmenterWidget.logic
        extents = self.extents
        for idx, id in enumerate(self.ids):
            lbl = self.segmentation.GetSegment(id).GetLabelValue()
            segment_arr = np.multiply(volume[extents[id]],(labelmap[extents[id]] == lbl),dtype=volume.dtype)
            segment_volume = slicer.util.addVolumeFromArray(segment_arr,name=f"{self.name}-{idx}")
            origin = logic.get_extent_origin_from_reference_volume(volumeNode,extents[id])
            segment_volume.SetOrigin(origin)
        return segment_volume
    def _create_segments(self,volume,labelmap):
        #self.segmentation.GetSegment(id).SetBinaryLabelmapRepresentationFromArray(array)
        # segment_arr = np.multiply(volume[self.extents[id]], (labelmap[self.extents[id]] == lbl), dtype=volume.dtype)
        pass
