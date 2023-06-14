import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *



#
# OtolithSegmenter
#

class OtolithSegmenter(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "OtolithSegmenter" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["Arthur Porto"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
      This module takes a volume and segments it using automated approaches. The output segments are converted to models.
      """
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
      This module was developed by Max (LastName), Arthur Porto and Adam P.Summers for the NSF-REU program at the University of Washington Friday Harbor Laboratories in 2023.
      """ # replace with organization, grant and thanks.

#
# OtolithSegmenterWidget
#

class OtolithSegmenterWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
  def onSelect(self):
    self.applyButton.enabled = bool(self.inputFile.currentPath)# and self.outputDirectory.currentPath

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    # Select input volume
    self.inputFile=ctk.ctkPathLineEdit()
    self.inputFile.filters = ctk.ctkPathLineEdit.Files
    self.inputFile.nameFilters = ["*.nii.gz"] 
    self.inputFile.setToolTip( "Select input volume" )
    parametersFormLayout.addRow("Input volume: ", self.inputFile)

    # Select output directory
    #self.outputDirectory=ctk.ctkPathLineEdit()
    #self.outputDirectory.filters = ctk.ctkPathLineEdit.Dirs
    #self.outputDirectory.setToolTip( "Select directory for output models: " )
    #parametersFormLayout.addRow("Output directory: ", self.outputDirectory)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Generate Otolith Segments."
    self.applyButton.enabled = True
    parametersFormLayout.addRow(self.applyButton)


    # connections
    self.inputFile.connect('validInputChanged(bool)', self.onSelect)
    #self.outputDirectory.connect('validInputChanged(bool)', self.onSelect)
    self.applyButton.connect('clicked(bool)', self.onApplyButton)

    # Add vertical spacer
    self.layout.addStretch(1)

  def cleanup(self):
    pass


  def onApplyButton(self):
    logic = OtolithSegmenterLogic()
    logic.run(self.inputFile.currentPath)#, self.outputDirectory.currentPath)

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
  def run(self, inputFile):#, outputDirectory):
    volumeNode = slicer.util.loadVolume(inputFile)
    voxelShrinkSize = 2


    # Create a new segmentation
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentationNode.CreateDefaultDisplayNodes() # only needed for display
    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
    addedSegmentID = segmentationNode.GetSegmentation().AddEmptySegment("otolith")

    # Create segment editor to get access to effects
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentEditorWidget.setSegmentationNode(segmentationNode)
    segmentEditorWidget.setSourceVolumeNode(volumeNode)


    # Apply Otsu thresholding
    segmentEditorWidget.setActiveEffectByName("Threshold")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("AutomaticThresholdMethod", "Otsu")
    effect.self().onApply()

    # Shrink the segment
    segmentEditorWidget.setActiveEffectByName("Margin")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("MarginSizeMm", -0.10)  # Adjust as needed
    effect.self().onApply()

    # Apply the islands effect
    segmentEditorWidget.setActiveEffectByName("Islands")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation", "SPLIT_ISLANDS_TO_SEGMENTS")
    effect.setParameter("MinimumSize", "1000")  # Adjust as needed
    effect.self().onApply()

    # Grow the segments back to their original size
    segmentEditorWidget.setActiveEffectByName("Margin")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("GrowFactor", 0.10)  # Adjust as needed
    effect.self().onApply()

    # Create a new model node for the segment
    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(
            slicer.mrmlScene)
    outputFolderId = shNode.CreateFolderItem(shNode.GetSceneItemID(),
                                                 'ModelsFolder')
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToModels(segmentationNode, outputFolderId)

    # Clean up
    segmentEditorWidget = None
    slicer.mrmlScene.RemoveNode(segmentationNode)



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
    pass
