import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from functools import partial
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import os

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
    self.parent.contributors = ["Arthur Porto", "Maximilian McKnight"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
      This module takes a volume and segments it using automated approaches. The output segments are converted to models.
      """
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
      This module was developed by Maximilian McKnight, Arthur Porto and Adam P.Summers for the NSF-REU program at the University of Washington Friday Harbor Laboratories in 2023.
      """ # replace with organization, grant and thanks.

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
    self.inputFile= ctk.ctkPathLineEdit()
    self.inputFile.filters = ctk.ctkPathLineEdit.Files
    self.inputFile.nameFilters = ["*.nii.gz"]
    self.inputFile.setToolTip( "Select input volume" )
    parametersFormLayout.addRow("Input volume: ", self.inputFile)
    
    #TODO: remove this for release
    # self.inputFile.currentPath = "/media/ap/Pocket/otho/test.nii.gz"
    self.inputFile.currentPath = "/home/max/Projects/fhl-work/holder/data/Otoliths/otoliths_raw/Holder 1 otolithJuanes1_13.8um_2k low res.nii.gz"
    
    # Select output directory
    self.outputDirectory=ctk.ctkPathLineEdit()
    self.outputDirectory.filters = ctk.ctkPathLineEdit.Dirs
    self.outputDirectory.setToolTip( "Select directory for output models: " )
    parametersFormLayout.addRow("Output directory: ", self.outputDirectory)

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

    print("hello world")
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
    apply_edit = partial(apply_segment_editor_effect, segmentEditorWidget)


    # Apply Otsu thresholding
    apply_edit(name = "Threshold", params = (("AutomaticThresholdMethod", "Otsu"),))

    # Shrink the segment
    apply_edit(name = "Margin", params = (("MarginSizeMm", -0.10),))

    # Apply the islands effect
    islandParams = (("Operation", "SPLIT_ISLANDS_TO_SEGMENTS"), ("MinimumSize", "1000"))
    apply_edit("Islands", islandParams)

    # Grow the segments back to their original size
    apply_edit(name="Margin", params=(("GrowFactor", 0.10),))

# PCA based clustering #TODO: pick/develop clustering algo

    # Get a list of segment IDs
    segmentNames = segmentationNode.GetSegmentation().GetSegmentIDs()

    # Get coordinates of each segment
    cords = np.array([(segmentationNode.GetSegmentCenterRAS(id)) for id in segmentNames])

    # Perform PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(cords)

    # Calculate the centroid of the structures in the original 3D space
    centroid = np.mean(pca_coords, axis=0)

    # Calculate the distance of each structure from the centroid
    distances = np.sqrt(np.sum((pca_coords - centroid)**2, axis=1))

    # Use 1/5 of the median distance as the size of the structure
    structure_size = np.median(distances)*2.2 / 5

    # Perform DBSCAN clustering to group structures that are in the same well
    dbscan = DBSCAN(eps=structure_size, min_samples=1)
    labels = dbscan.fit_predict(cords)

    # Pair each group with its distance from the overall centroid and sort the pairs by distance
    group_cords = np.array([np.mean(pca_coords[labels == label], axis=0) for label in np.unique(labels)])
    group_distances = np.sqrt(np.sum((group_cords - centroid)**2, axis=1))
    pairs = sorted(enumerate(group_distances), key=lambda pair: pair[1], reverse=True)

    # For each group, calculate the angle with respect to PC1
    group_angles = np.arctan2(group_cords[:,1], group_cords[:,0])

    # Cluster the groups into two rows based on their distance from the overall centroid
    kmeans = KMeans(n_clusters=2, n_init=10)
    row_labels = kmeans.fit_predict(group_distances.reshape(-1, 1))

    # Sort the groups by row and then by angle within each row
    outer_row_indices = [i for i, label in enumerate(row_labels) if label == 0]
    inner_row_indices = [i for i, label in enumerate(row_labels) if label == 1]
    outer_row_indices.sort(key=lambda i: group_angles[i])
    inner_row_indices.sort(key=lambda i: group_angles[i])
    sorted_group_indices = outer_row_indices + inner_row_indices
   
    
  # simple NN clustering 

    # #plan - get centers. get nearest vertical neighbor, group those. print those groups. name them.
    # segmentNames = segmentationNode.GetSegmentation().GetSegmentIDs() # get a list of segment names to reference segements
    # cords = [(segmentationNode.GetSegmentCenterRAS(id)) for id in segmentNames] # get cords of each segment
    # distTree = sci.spatial.KDTree(cords) #create a tree to calulate nearest segments
    # nearest = distTree.query(cords, k = 2)[1][:,1] #get a list of nearest neighbor pairs
    # pairs = enumerate(nearest) #pair each node with its nearest neighbor
    # uniquePairs = {tuple(sorted(pair)) for pair in pairs} # remove duplicate pairs
    # print(pairs)
    # print(uniquePairs)

    # Export node TODO: review and potential cleanup/extract into function
    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    outputFolderId = shNode.CreateFolderItem(shNode.GetSceneItemID(), 'ModelsFolder')

    for index, group_index in enumerate(sorted_group_indices):
        holder_name = f"holder-{index}"
        folder = shNode.CreateFolderItem(outputFolderId, holder_name) # create a folder for each holder
        segment_ids_in_group = [segmentNames[i] for i, label in enumerate(labels) if label == group_index]
        for segment_index, segment_id in enumerate(segment_ids_in_group):
            model_name = f"{holder_name}-model-{segment_index}"
            slicer.modules.segmentations.logic().ExportSegmentsToModels(segmentationNode, [segment_id], folder)
            model_node = slicer.mrmlScene.GetNthNodeByClass(slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLModelNode') - 1, 'vtkMRMLModelNode')
            if model_node is not None:
                model_node.SetName(model_name)
                model_directory = os.path.join(outputDirectory, holder_name)
                os.makedirs(model_directory, exist_ok=True)
                # Save the model node to the directory
                slicer.util.saveNode(model_node, os.path.join(model_directory, model_name + ".ply"))
        shNode.SetItemParent(folder,outputFolderId) # for some reason ExportSegmentsToModels undoes nesting of a node so we need to renest it
    #Figure out how to export subject heirarchy parent folderid

    # slicer.modules.segmentations.logic().ExportVisibleSegmentsToModels(segmentationNode, outputFolderId)

    # Clean up
    segmentEditorWidget = None
    # slicer.mrmlScene.RemoveNode(segmentationNode) #TODO: uncomment after dev finished



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
    #print(params)
    for param in params:
      #print(type(param))
      effect.setParameter(*param)
    effect.self().onApply()
    return effect
