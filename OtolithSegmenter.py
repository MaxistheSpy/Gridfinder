import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from functools import partial
import numpy as np
import scipy
import os

slicer.util.pip_install('imutils')
slicer.util.pip_install('connected-components-3d')
slicer.util.pip_install('fastremap')
slicer.util.pip_install('scikit-image')
import cc3d
import fastremap
from Otolith_segmenter_utils.gridfinder_finalized import pair_otoliths_to_grid
import Otolith_segmenter_utils.env_paths as env
from Otolith_segmenter_utils.general_utils import timer
from Otolith_segmenter_utils.holder import Holder
from Otolith_segmenter_utils.well import Well
from skimage import color


#
# OtolithSegmenter
#

class OtolithSegmenter(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
      https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
      """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "OtolithSegmenter"  # TODO make this more human readable by adding spaces
        self.parent.categories = ["Examples"]
        self.parent.dependencies = ['OpenCV']
        self.parent.contributors = ["Arthur Porto",
                                    "Maximilian McKnight"]  # replace with "Firstname Lastname (Organization)"
        self.parent.helpText = """
      This module takes a volume and segments it using automated approaches. The output segments are converted to models.
      """
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = """
      This module was developed by Maximilian McKnight, Arthur Porto and Adam P.Summers for the NSF-REU program at the University of Washington Friday Harbor Laboratories in 2023.
      """  # replace with organization, grant and thanks.


#
# OtolithSegmenterWidget
#

class OtolithSegmenterWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
      https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
      """

    def onSelect(self):
        self.applyButton.enabled = bool(self.inputFile.currentPath and self.outputDirectory.currentPath)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # print(self.inputFile,"t")
        # Parameters Area
        #
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)

        # Layout within the dummy collapsible button
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        # Select input volume
        self.inputFile = ctk.ctkPathLineEdit()
        self.inputFile.filters = ctk.ctkPathLineEdit.Files
        self.inputFile.nameFilters = ["*.nii.gz"]
        self.inputFile.setToolTip("Select input volume")
        parametersFormLayout.addRow("Input volume: ", self.inputFile)

        # TODO: remove preset paths for release
        self.inputFile.currentPath = env.TESTPATH_IMPORT_FILE

        # Select output directory
        self.outputDirectory = ctk.ctkPathLineEdit()
        self.outputDirectory.filters = ctk.ctkPathLineEdit.Dirs
        self.outputDirectory.setToolTip("Select directory for output models: ")
        parametersFormLayout.addRow("Output directory: ", self.outputDirectory)
        self.outputDirectory.currentPath = env.TESTPATH_OUTPUT_FILE

        #
        # Apply Button
        #
        self.applyButton = qt.QPushButton("Apply")
        self.applyButton.toolTip = "Generate Otolith Segments."
        self.applyButton.enabled = True
        parametersFormLayout.addRow(self.applyButton)

        #
        # Import Button
        #
        self.importButton = qt.QPushButton("Import")
        self.importButton.toolTip = "Import Input Volume."
        self.importButton.enabled = True
        parametersFormLayout.addRow(self.importButton)

        #
        # Segment Button
        #
        self.segmentButton = qt.QPushButton("Segment")
        self.segmentButton.toolTip = "Segment Volume"
        self.segmentButton.enabled = True
        parametersFormLayout.addRow(self.segmentButton)

        #
        # Export Button
        #
        self.exportButton = qt.QPushButton("Export")
        self.exportButton.toolTip = "Export Holder"
        self.exportButton.enabled = True
        parametersFormLayout.addRow(self.exportButton)

        # connections
        self.inputFile.connect('validInputChanged(bool)', self.onSelect)
        self.outputDirectory.connect('validInputChanged(bool)', self.onSelect)
        self.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.importButton.connect('clicked(bool)', self.onImportButton)
        self.segmentButton.connect('clicked(bool)', self.onSegmentButton)
        self.exportButton.connect('clicked(bool)', self.onExportButton)
        # Define Instance Variables
        self.volume = None
        self.segmentation = None
        self.logic = OtolithSegmenterLogic()
        self.holder = None
        self.extents = None
        self.mask= None
        self.holder = None
        self.volume_list = None

        # Add vertical spacer
        self.layout.addStretch(1)

    def cleanup(self):
        # slicer.mrmlScene.RemoveNode(self.segmentation)
        # slicer.mrmlScene.RemoveNode(self.volume)
        self.volume = None

    def onApplyButton(self):
        with slicer.util.RenderBlocker():
            with timer("full run"):
                if self.volume is None:
                    self.onImportButton()
                if self.volume is not None and self.segmentation is None:
                    self.segmentation,self.extents,self.mask = self.logic.segment(self.volume, slicer.mrmlScene)
                self.holder = self.logic.run(self.inputFile.currentPath, self.outputDirectory.currentPath,
                                             self.volume, self.segmentation,self.extents,self.mask)
    #TODO: Investigate render blocker
    #with slicer.util.RenderBlocker():
    def onImportButton(self):
        with slicer.util.RenderBlocker():
            if self.volume == None:
                self.volume = self.logic.importVolume(self.inputFile.currentPath)

    def onSegmentButton(self):
        with slicer.util.RenderBlocker():
            if self.volume is None:
                self.onImportButton()
            if self.volume is not None and self.segmentation is None:
                self.segmentation,self.extents,self.mask = self.logic.segment(self.volume, slicer.mrmlScene)
    def onExportButton(self):
        with slicer.util.RenderBlocker():
            if self.holder is None:
                self.onApplyButton()
            with timer("export"):
                self.volume_list = self.holder.export_wells_to_volume_list(self.mask)


#
# OtolithSegmenterLogic
#

class OtolithSegmenterLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
      computation done by your module.  The interface
      should be such that other python code can import
      this class and make use of the functionality without
      requiring an instance of the Widget.
      Uses ScriptedLoadableModuleLogic base class, available at:
      https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
      """

    def importVolume(self, inputFile):
        return slicer.util.loadVolume(inputFile)

    @staticmethod
    def IJK_to_RAS_points(RAS_points, volumeNode):
        logic = OtolithSegmenterLogic()
        IJK_matrix = logic.get_IJK_matrix(volumeNode)
        translation = logic.get_translation_node(volumeNode)

        if translation:
            RAS_points = [translation.TransformPoint(point[0:3]) for point in RAS_points]

        RAS_points = np.hstack((RAS_points, np.ones(RAS_points.shape[0]).reshape(-1, 1))).T
        transformed_points = np.matmul(IJK_matrix, RAS_points)
        transformed_points = transformed_points.T[:, :3].astype(int)
        return transformed_points

    @staticmethod
    def get_IJK_matrix(volumeNode):
        """
        Takes in a volume node and gets the RAS to IJK matrix associated with that node and returns it as a numpy array
        """

        RAS_to_IJK = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(RAS_to_IJK)
        RAS_to_IJK_np = slicer.util.arrayFromVTKMatrix(RAS_to_IJK)
        return RAS_to_IJK_np

    @staticmethod
    def IJK_to_RAS_point(RAS_point, volumeNode):
        logic = OtolithSegmenterLogic()
        RAS_to_IJK = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(RAS_to_IJK)
        translation = logic.get_translation_node(volumeNode)
        if translation:
            RAS_point = translation.MultiplyPoint(RAS_point)
        RAS_point = np.append(RAS_point, 1)
        transformed_point = RAS_to_IJK.MultiplyPoint(RAS_point)
        return transformed_point

    @staticmethod
    def get_translation_node(volumeNode):
        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        parent_transform = volumeNode.GetParentTransformNode()
        if parent_transform is None:
            return False
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, parent_transform, transformRasToVolumeRas)
        transform_node = transformRasToVolumeRas
        return transform_node
    @staticmethod
    def get_extent_origin_from_reference_volume(volumeNode,extent):
        print(extent)
        # print([x for x in extent])
        point = [x.start for x in extent]
        point_in_KJI = point[::-1]
        #NOTE: We Need to reverse point because of fortran vs c ordering, A more elegant fix probably exists
        RAS_to_IJK = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(RAS_to_IJK)
        origin = RAS_to_IJK.MultiplyPoint(point_in_KJI+[1])
        print("extent:",extent,"\norigin:",origin,"\npoint:",point,"\npoint in KJI:",point_in_KJI)
        return origin[:3]

    @staticmethod
    def segment(volumeNode, scene, threshold=63, larger_than=1000, connectivity=26):
        # Create a new segmentation Node
        segmentationNode = scene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.CreateDefaultDisplayNodes()  # only needed for display
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        # Create labelmap node
        labelmapVolumeNode = scene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
        # export volumeNode to numpy
        volume_array = slicer.util.arrayFromVolume(volumeNode)
        # threshold volume
        # label connected components

        # TODO: Improve memory useage with CC3D
        mask, n = cc3d.connected_components(volume_array > threshold, connectivity=connectivity, return_N=True)
        voxels, extents, centroids = cc3d.statistics(mask).values()

        # TODO: verify this wont accidentally remove a segment
        # NOTE: this code should remove the segment calculated from 0 which will normally be largest so we want to toss it
        voxels_sans_zero = voxels
        voxels_sans_zero[0] = 0
        print(voxels_sans_zero[:50])
        valid_idxs = np.argwhere(voxels_sans_zero > larger_than)
        if valid_idxs == []:
            print("No valid segments found")
            return
        ranks = scipy.stats.rankdata(-voxels[valid_idxs], method='ordinal')
        # create dict to remap labels
        arr = np.zeros(len(voxels))
        # print(type(valid_idxs))
        # print(valid_idxs)
        valid_idxs = valid_idxs.flatten().tolist()
        print(valid_idxs)
        for rank_idx, list_idx in enumerate(valid_idxs):
            arr[list_idx] = ranks[rank_idx]
        dict = {idx: rank for idx, rank in enumerate(arr.tolist())}
        # remap labels
        mask = fastremap.remap(mask, dict, in_place=True)
        # import npy array to labelmap
        slicer.util.updateVolumeFromArray(labelmapVolumeNode, mask)
        labelmapVolumeNode.SetOrigin(volumeNode.GetOrigin())
        segment_extents = {f"Label_{int(dict[idx])}": extents[idx] for idx in valid_idxs}
        print("segments:", segment_extents)
        # import labelmap to segmentation node
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, segmentationNode)

        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
        return segmentationNode, segment_extents,mask

    def run(self, inputFile, outputDirectory, volumeNode=None, segmentationNode=None,extents=None,mask=None):
        logic = OtolithSegmenterLogic()
        print("hello world")
        if volumeNode is None:
            volumeNode = slicer.util.loadVolume(inputFile)
        voxelShrinkSize = 2
        scene = slicer.mrmlScene

        # Get a list of segment IDs

        # segmentationNode = slicer.util.loadSegmentation(inputFile) #Testing?
        segmentIDs = segmentationNode.GetSegmentation().GetSegmentIDs()
        # TODO: this is probably sub optimal as it may fail if segment names get changed
        #tie each segment to its extent

        # Get coordinates of each segment
        # volume_array = slicer.util.arrayFromVolume(volumeNode)

        # Get segment coords and convert to IJK
        cords = np.array([(segmentationNode.GetSegmentCenterRAS(segment)) for segment in segmentIDs])
        IJK_cords = logic.IJK_to_RAS_points(cords, volumeNode)
        # find wells from IJK cords
        # well_list, _, _ = pair_otoliths_to_grid(slicer.util.arrayFromVolume(volumeNode), IJK_cords)  # TODO: Refactor to simplify output
        well_list = list(map(list, list(np.load(env.TESTPATH_QUICKLOAD_WELLS))))
        print(well_list)
        print(segmentIDs)

        holder = Holder(segmentIDs, well_list,extents,segmentationNode, volumeNode)
        return holder
        # TODO: create export feature
        # use lab colors to color segments
    def export_wells(self,holder,mask):
        pass

        # TODO: Look at this code and see if theres anything useful

        # # Export node
        # shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(scene)
        # outputFolderId = shNode.CreateFolderItem(shNode.GetSceneItemID(), 'ModelsFolder')
        #
        # def export_to_model(segment, folder):
        #     slicer.modules.segmentations.logic().ExportSegmentsToModels(segmentationNode, [segment], folder)
        #     otolith = scene.GetNthNodeByClass(scene.GetNumberOfNodesByClass('vtkMRMLModelNode') - 1,
        #                                             'vtkMRMLModelNode')
        #     return otolith
        #
        # well_contents_mapped = [segmentNames[idx] for well in wells for idx in well]
        # print(well_contents_mapped)
        # #TODO: this is doing two things, exporting nodes to internal path and external path, fix
        #
        # for index, well_contents in enumerate(well_contents_mapped):
        #
        #     well_name = f"well-{index}"
        #
        #     # create relevant folder in slicer and on system path for well
        #     well_folder = shNode.CreateFolderItem(outputFolderId, well_name)
        #     model_path = os.path.join(outputDirectory, well_name)
        #     os.makedirs(model_path, exist_ok=True)
        #     print(well_contents)
        #
        #     for segment_index, segment in enumerate(well_contents):
        #         print(segment_index,segment)
        #         otolith_node = export_to_model(segment, well_folder)
        #         print(otolith_node)
        #         if otolith_node is None:
        #             continue
        #
        #         otolith_node.SetName(f"{well_name}-model-{segment_index}")
        #         slicer.util.saveNode(otolith_node, os.path.join(model_path, segment + ".ply"))
        #     shNode.SetItemParent(well_folder, outputFolderId)  # ExportSegmentsToModels undoes nesting of a node
        # Figure out how to export subject heirarchy parent folderid
        # slicer.modules.segmentations.logic().ExportVisibleSegmentsToModels(segmentationNode, outputFolderId)

        # # Clean up #TODO: Write cleanup function when done
        # segmentEditorWidget = None
        # scene.RemoveNode(segmentationNode)


class OtolithSegmenterTest(ScriptedLoadableModuleTest):
    """
      This is the test case for your scripted module.
      Uses ScriptedLoadableModuleTest base class, available at:
      https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
      """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
          """
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run as few or as many tests as needed here.
          """
        self.setUp()
        self.test_OtolithSegmenter1()

    def test_OtolithSegmenter1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
          tests should exercise the functionality of the logic with different inputs
          (both valid and invalid).  At higher levels your tests should emulate the
          way the user would interact with your code and confirm that it still works
          the way you intended.
          One of the most important features of the tests is that it should alert other
          developers when their changes will have an impact on the behavior of your
          module.  For example, if a developer removes a feature that you depend on,
          your test should break so they know that the feature is needed.
          """
        slicer.util.selectModule('OtolithSegmenter')
        pass


def apply_segment_editor_effect(widget, name: str, params: tuple):
    widget.setActiveEffectByName(name)
    effect = widget.activeEffect()
    # print(params)
    for param in params:
        # print(type(param))
        effect.setParameter(*param)
    effect.self().onApply()
    return effect
