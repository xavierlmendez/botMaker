class TreeNode:
    def __init__(self, parentNode = None, data = None, children = None):
        self.parentNode = parentNode
        self.data = data # leaving abstract here to allow more options in the decision tree implementations
        self.children = children
        pass
    
