import numpy as np

class GraphEdge:
    def __init__(self):
        pass
    
    # todo figure out how I want to handle the relation ship and responsibilites of the edge versus node classes

class GraphNode:
    def __init__(self, nodeIdentifier = 0, data = None, edges=None):
        self.identifier = nodeIdentifier
        self.data = data # leaving abstract here to allow more options in graph implementations

        if edges is None:
            edges = []
            self.edgeCount = 0
        else :
            self.edges = edges
            self.edgeCount = len(edges)

    def addEdge(self, GraphNode:object):
        self.edgeCount += 1
        
        self.edges.append(GraphNode)

    def removeEdge(self, GraphNode:object):
        self.edgeCount -= 1
        self.edges.remove(GraphNode)
        GraphNode.parentNode = None # if not set on init then this will correct


class Graph:
    def __init__(self, nodes = None, useCustomIdentifier = False):
        self.identifierIncrementor = 0;
        
        pass

