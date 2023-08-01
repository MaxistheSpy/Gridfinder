import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from functools import partial
import numpy as np
import cv2
import os
slicer.util.pip_install('imutils')
from Otolith_segmenter_utils.gridfinder_finalized import pair_otoliths_to_grid
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

        # TODO: remove this for release
        # self.inputFile.currentPath = "/media/ap/Pocket/otho/test.nii.gz"
        self.inputFile.currentPath = "/home/max/Projects/test/cod2_27um_2k_top.nii.gz"

        # Select output directory
        self.outputDirectory = ctk.ctkPathLineEdit()
        self.outputDirectory.filters = ctk.ctkPathLineEdit.Dirs
        self.outputDirectory.setToolTip("Select directory for output models: ")
        parametersFormLayout.addRow("Output directory: ", self.outputDirectory)
        self.outputDirectory.currentPath = "/home/max/Projects/test/junk"

        #
        # Apply Button
        #
        self.applyButton = qt.QPushButton("Apply")
        self.applyButton.toolTip = "Generate Otolith Segments."
        self.applyButton.enabled = True
        parametersFormLayout.addRow(self.applyButton)

        # connections
        self.inputFile.connect('validInputChanged(bool)', self.onSelect)
        self.outputDirectory.connect('validInputChanged(bool)', self.onSelect)
        self.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Add vertical spacer
        self.layout.addStretch(1)

    def cleanup(self):
        pass

    def onApplyButton(self):
        logic = OtolithSegmenterLogic()
        logic.run(self.inputFile.currentPath, self.outputDirectory.currentPath)


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

    def run(self, inputFile, outputDirectory):
        def IJK_to_RAS_points(RAS_points, volumeNode):
            IJK_matrix = get_IJK_matrix(volumeNode)
            translation = get_translation_node(volumeNode)

            if translation:
                RAS_points = [translation.TransformPoint(point[0:3]) for point in RAS_points]

            RAS_points = np.hstack((RAS_points, np.ones(RAS_points.shape[0]).reshape(-1, 1))).T
            transformed_points = np.matmul(IJK_matrix, RAS_points)
            transformed_points = transformed_points.T[:,:3].astype(int)
            return transformed_points

        def get_translation_node(volumeNode):
            transformRasToVolumeRas = vtk.vtkGeneralTransform()
            parent_transform = volumeNode.GetParentTransformNode()
            if parent_transform is None:
                return False
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, parent_transform, transformRasToVolumeRas)
            transform_node = transformRasToVolumeRas
            return transform_node

        def get_IJK_matrix(volumeNode):
            RAS_to_IJK = vtk.vtkMatrix4x4()
            volumeNode.GetRASToIJKMatrix(RAS_to_IJK)
            RAS_to_IJK_np = slicer.util.arrayFromVTKMatrix(RAS_to_IJK)
            return RAS_to_IJK_np
        def IJK_to_RAS_points_alt(RAS_points, volumeNode):
            [IJK_to_RAS_point(RAS_point, volumeNode) for RAS_point in RAS_points]
        def IJK_to_RAS_point(RAS_point, volumeNode):
            RAS_to_IJK = vtk.vtkMatrix4x4()
            volumeNode.GetRASToIJKMatrix(RAS_to_IJK)
            translation = get_translation_node(volumeNode)
            if translation:
                RAS_point = translation.MultiplyPoint(RAS_point)
            RAS_point = np.append(RAS_point, 1)
            transformed_point = RAS_to_IJK.MultiplyPoint(RAS_point)
            return transformed_point

        print("hello world")
        volumeNode = slicer.util.loadVolume(inputFile)
        voxelShrinkSize = 2
        scene = slicer.mrmlScene

        # Create a new segmentation
        segmentationNode = scene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.CreateDefaultDisplayNodes()  # only needed for display
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        addedSegmentID = segmentationNode.GetSegmentation().AddEmptySegment("otolith")

        # Create segment editor to get access to effects
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(scene)
        segmentEditorNode = scene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
        segmentEditorWidget.setSegmentationNode(segmentationNode)
        segmentEditorWidget.setSourceVolumeNode(volumeNode)
        apply_edit = partial(apply_segment_editor_effect, segmentEditorWidget)

        # Apply Otsu thresholding
        apply_edit(name="Threshold", params=(("AutomaticThresholdMethod", "Otsu"),))

        # Shrink the segment
        apply_edit(name="Margin", params=(("MarginSizeMm", -0.10),))

        # Apply the islands effect
        islandParams = (("Operation", "SPLIT_ISLANDS_TO_SEGMENTS"), ("MinimumSize", "1000"))
        apply_edit("Islands", islandParams)

        # Grow the segments back to their original size
        apply_edit(name="Margin", params=(("GrowFactor", 0.10),))

        # PCA based clustering #TODO: pick/develop clustering algo

        # Get a list of segment IDs
        segmentNames = segmentationNode.GetSegmentation().GetSegmentIDs()

        # Get coordinates of each segment
        volume_array = slicer.util.arrayFromVolume(volumeNode)
        cords = np.array([(segmentationNode.GetSegmentCenterRAS(segment)) for segment in segmentNames])



        IJK_cords = IJK_to_RAS_points(cords, volumeNode)
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
