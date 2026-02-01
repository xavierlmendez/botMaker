from dataclasses import dataclass, field
from typing import Any, Set

# https://dzone.com/articles/understanding-pythons-dataclass-decorator
@dataclass(frozen=True, slots=True)
class Edge:
    u: int
    v: int
    data: Any = None

@dataclass(slots=True)
class GraphNode:
    nodeId: int
    data: Any
    neighbors: Set[int] = field(default_factory=set)

class Graph:
    def __init__(self, initNodes = None):
        self.nodes: dict[int, GraphNode] = {}
        self.edges = [] # (Bi)directional edges will have to be on a directional graph implementation
        self.idIncrementor = 0 # Prefer this as searching burned identifiers will add to run time

        if initNodes is not None:
            for nodeId in initNodes:
                self.nodes[nodeId] = initNodes[nodeId]
        
    def addNode(self, data = None):
        self.idIncrementor += 1
        nodeId = self.idIncrementor

        newNode = GraphNode(nodeId, data)
        self.nodes[nodeId] = newNode
        return nodeId
        
    def addEdge(self, nodeIdOne, nodeIdTwo):
        if nodeIdOne not in self.nodes:
            raise KeyError(f"Unknown node: {nodeIdOne}")
        if nodeIdTwo not in self.nodes:
            raise KeyError(f"Unknown node: {nodeIdTwo}")

        newEdge = Edge(nodeIdOne, nodeIdTwo)
        if newEdge in self.edges:
            raise KeyError(f"Duplicate edge: {newEdge}")

        self.edges.append(newEdge)
        self.nodes[nodeIdOne].neighbors.add(nodeIdTwo)
        self.nodes[nodeIdTwo].neighbors.add(nodeIdOne)

    def removeNode(self, nodeId):
        if nodeId not in self.nodes:
            raise KeyError(f"Unknown node: {nodeId}")

        for connectedNodeId in self.nodes[nodeId].neighbors:
            self.nodes[connectedNodeId].neighbors.remove(nodeId)

        del self.nodes[nodeId]

