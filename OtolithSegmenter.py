import fastremap
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from functools import partial
import numpy as np
import os
slicer.util.pip_install('imutils')
slicer.util.pip_install('connected-components-3d')
slicer.util.pip_install('fastremap')
from Otolith_segmenter_utils.fastsegment import get_segments_from_matrix
from Otolith_segmenter_utils.gridfinder_finalized import pair_otoliths_to_grid
import Otolith_segmenter_utils.env_paths as env
import cc3d
from Otolith_segmenter_utils.general_utils import timer
from Otolith_segmenter_utils import fastsegment
import scipy
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

        # connections
        self.inputFile.connect('validInputChanged(bool)', self.onSelect)
        self.outputDirectory.connect('validInputChanged(bool)', self.onSelect)
        self.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.importButton.connect('clicked(bool)', self.onImportButton)
        self.segmentButton.connect('clicked(bool)', self.onSegmentButton)
        # Define Instance Variables
        self.volume = None
        self.segmentation = None
        self.logic = OtolithSegmenterLogic()

        # Add vertical spacer
        self.layout.addStretch(1)

    def cleanup(self):
        slicer.mrmlScene.RemoveNode(self.volume)
        self.volume = None

    def onApplyButton(self):
        return
        if self.volume is not None:
            self.logic.run(self.inputFile.currentPath, self.outputDirectory.currentPath, self.volume)
        else:
            self.logic.run(self.inputFile.currentPath, self.outputDirectory.currentPath)

    def onImportButton(self):
        if self.volume == None:
            self.volume = self.logic.importVolume(self.inputFile.currentPath)
    def onSegmentButton(self):
        if self.volume is not None:
            with timer("segment"):
                self.segmentation = self.logic.segment(self.volume,slicer.mrmlScene)
        else:
            self.onImportButton()
            self.onSegmentButton()

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
    def segment(volumeNode,scene,threshold=63,larger_than=1000,connectivity=26):
        # Create a new segmentation Node
        segmentationNode = scene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.CreateDefaultDisplayNodes() # only needed for display
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        # Create labelmap node
        labelmapVolumeNode = scene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
        # export volumeNode to numpy
        volume_array = slicer.util.arrayFromVolume(volumeNode)
        # threshold volume
        # label connected components

        #TODO: Improve memory useage with CC3D
        mask, n = cc3d.connected_components(volume_array > threshold, connectivity=connectivity, return_N=True)
        voxels, extents, centroids = cc3d.statistics(mask).values()

        #TODO: verify this wont accidentally remove a segment
        #NOTE: this code should remove the segment calculated from 0 which will normally be largest so we want to toss it
        voxels_sans_zero = voxels
        voxels_sans_zero[0] = 0
        print(voxels_sans_zero[:50])
        valid_idxs = np.argwhere(voxels_sans_zero > larger_than)
        if valid_idxs == []:
            print("No valid segments found")
            return
        ranks = scipy.stats.rankdata(-voxels[valid_idxs], method='ordinal')
        #create dict to remap labels
        arr = np.zeros(len(voxels))
        for rank_idx,list_idx in enumerate(valid_idxs):
            arr[list_idx] = ranks[rank_idx]
        dict = {idx:rank for idx,rank in enumerate(arr)}
        #remap labels
        mask = fastremap.remap(mask,dict,in_place=True)
        # import npy array to labelmap
        slicer.util.updateVolumeFromArray(labelmapVolumeNode, mask)
        labelmapVolumeNode.SetOrigin(volumeNode.GetOrigin())

        # import labelmap to segmentation node
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, segmentationNode)
        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
        return segmentationNode




    def run(self, inputFile, outputDirectory, volumeNode=None):
        logic = OtolithSegmenterLogic()
        print("hello world")
        if volumeNode is None:
            volumeNode = slicer.util.loadVolume(inputFile)
        voxelShrinkSize = 2
        scene = slicer.mrmlScene

        ##### OLD SEGMENTATION METHOD #####
        # # Create a new segmentation
        # segmentationNode = scene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        # segmentationNode.CreateDefaultDisplayNodes()  # only needed for display
        # segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        # addedSegmentID = segmentationNode.GetSegmentation().AddEmptySegment("otolith")
        #
        # # Create segment editor to get access to effects
        # segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        # segmentEditorWidget.setMRMLScene(scene)
        # segmentEditorNode = scene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        # segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
        # segmentEditorWidget.setSegmentationNode(segmentationNode)
        # segmentEditorWidget.setSourceVolumeNode(volumeNode)
        # apply_edit = partial(apply_segment_editor_effect, segmentEditorWidget)
        #
        # # Apply Otsu thresholding
        # apply_edit(name="Threshold", params=(("AutomaticThresholdMethod", "Otsu"),))
        #
        # # Shrink the segment
        # apply_edit(name="Margin", params=(("MarginSizeMm", -0.10),))
        #
        # # Apply the islands effect
        # islandParams = (("Operation", "SPLIT_ISLANDS_TO_SEGMENTS"), ("MinimumSize", "1000"))
        # apply_edit("Islands", islandParams)
        #
        # # Grow the segments back to their original size
        # apply_edit(name="Margin", params=(("GrowFactor", 0.10),))
        ##### OLD SEGMENTATION METHOD #####


        # Get a list of segment IDs

        segmentationNode = slicer.util.loadSegmentation(inputFile)
        segmentNames = segmentationNode.GetSegmentation().GetSegmentIDs()

        # Get coordinates of each segment
        volume_array = slicer.util.arrayFromVolume(volumeNode)
        cords = np.array([(segmentationNode.GetSegmentCenterRAS(segment)) for segment in segmentNames])



        IJK_cords = logic.IJK_to_RAS_points(cords, volumeNode)
        wells, _, _ = pair_otoliths_to_grid(volume_array, IJK_cords)
        print(wells)
        print(segmentNames)



        # Export node TODO: review and potential cleanup/extract into function
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(scene)
        outputFolderId = shNode.CreateFolderItem(shNode.GetSceneItemID(), 'ModelsFolder')

        def export_to_model(segment, folder):
            slicer.modules.segmentations.logic().ExportSegmentsToModels(segmentationNode, [segment], folder)
            otolith = scene.GetNthNodeByClass(scene.GetNumberOfNodesByClass('vtkMRMLModelNode') - 1,
                                                    'vtkMRMLModelNode')
            return otolith

        well_contents_mapped = [segmentNames[idx] for well in wells for idx in well]
        print(well_contents_mapped)
        #TODO: this is doing two things, exporting nodes to internal path and external path, fix

        for index, well_contents in enumerate(well_contents_mapped):

            well_name = f"well-{index}"

            # create relevant folder in slicer and on system path for well
            well_folder = shNode.CreateFolderItem(outputFolderId, well_name)
            model_path = os.path.join(outputDirectory, well_name)
            os.makedirs(model_path, exist_ok=True)
            print(well_contents)

            for segment_index, segment in enumerate(well_contents):
                print(segment_index,segment)
                otolith_node = export_to_model(segment, well_folder)
                print(otolith_node)
                if otolith_node is None:
                    continue

                otolith_node.SetName(f"{well_name}-model-{segment_index}")
                slicer.util.saveNode(otolith_node, os.path.join(model_path, segment + ".ply"))
            shNode.SetItemParent(well_folder, outputFolderId)  # ExportSegmentsToModels undoes nesting of a node

        # Figure out how to export subject heirarchy parent folderid

        # slicer.modules.segmentations.logic().ExportVisibleSegmentsToModels(segmentationNode, outputFolderId)
        #
        # # Clean up
        # segmentEditorWidget = None
        # scene.RemoveNode(segmentationNode) #TODO: uncomment after dev finished


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
