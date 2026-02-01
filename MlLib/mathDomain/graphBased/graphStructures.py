import numpy as np
class GraphNode:
    def __init__(self, nodeIdentifier = 0, data = None, edges=None):
        self.nodeId = nodeIdentifier
        self.data = data # leaving abstract here to allow more options in graph implementations

        if edges is None:
            self.outboundEdges = []
            self.edgeCount = 0
        else :
            self.outboundEdges = edges
            self.edgeCount = len(edges)

    def addToEdge(self, graphNode:object):
        self.edgeCount += 1
        self.outboundEdges.append(graphNode.nodeId)
        
    def removeEdge(self, graphNode:object):
        self.edgeCount -= 1
        self.outboundEdges.remove(graphNode.nodeId)


class Graph:
    def __init__(self, nodes = None, useCustomIdentifier = False):
        
        self.nodes = []
        self.nodeCount = 0
        self.edges = [] # (Bi)directional edges will have to be on a directional graph implementation
        self.useCustomIdentifier = useCustomIdentifier
        self.identifierIncrementor = 0 # Prefer this as searching burned identifiers will add to run time
        self.burnedIdentifiers = []
        
        # todo implement handling for nodes passed into constructor
        
    def addNode(self, data = None, customIdentifier = None):
        try:
            if(self.useCustomIdentifier and customIdentifier is not None):
                nodeId = customIdentifier
            else:
                self.identifierIncrementor += 1
                nodeId = self.identifierIncrementor
                
            newNode = GraphNode(nodeId, data)
            self.nodes.append(newNode)
            self.nodeCount += 1
            
        except:
            # todo figure out how I want to handle errors for mathDomain
            pass
        
    def addEdge(self, nodeIdOne, nodeIdTwo):
        # confirm nodes exist
        # add edge if so 
        # else pop up an error
        pass
    def removeNode(self):
        try:
            pass
            # need to remove node from all edges
            # remove from node list
            # decrement node count
        
        except:
        # todo figure out how I want to handle errors for mathDomain
            pass

