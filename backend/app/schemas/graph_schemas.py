from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class EntityNode(BaseModel):
    entity_id: str
    name: str
    type: str           # PERSON, ORG, CONCEPT, LOCATION, etc.
    description: Optional[str] = None
    doc_ids: List[str] = []
    mention_count: int = 0


class RelationshipEdge(BaseModel):
    source_entity: str
    target_entity: str
    relationship_type: str
    weight: float = 1.0
    doc_id: Optional[str] = None


class GraphResponse(BaseModel):
    nodes: List[EntityNode]
    edges: List[RelationshipEdge]
    total_nodes: int
    total_edges: int


class TraversalRequest(BaseModel):
    start_entity: str
    max_depth: int = Field(default=2, ge=1, le=5)
    relationship_types: Optional[List[str]] = None  # None = all types


class TraversalResponse(BaseModel):
    start_entity: str
    depth_reached: int
    nodes: List[EntityNode]
    edges: List[RelationshipEdge]
    paths: List[List[str]]  # list of entity name paths found
