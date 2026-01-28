import numpy as np
class TreeNode:
    def __init__(self, parentNode = None, data = None, children: np.ndarray = None):
        self.parentNode = parentNode
        self.data = data # leaving abstract here to allow more options in the decision tree and other tree implementations
        self.childNodes = []
        self.childNodeCount = 0
        self.isLeafNode = False

    def addChild(self, child:object):
        self.childNodeCount += 1
        self.childNodes.append(child)
        child.parentNode = self # if not set on init then this will correct

    def removeChild(self, child:object):
        self.childNodeCount -= 1
        self.childNodes.remove(child)
        child.parentNode = None # if not set on init then this will correct