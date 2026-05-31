from fastapi import APIRouter, HTTPException
from backend.app.schemas.graph_schemas import (
    GraphResponse,
    TraversalRequest,
    TraversalResponse,
)
from backend.retrieval.graph_retriever import (
    get_all_entities,
    get_all_relationships,
    traverse_from_entity,
)

router = APIRouter()


@router.get("/explore", response_model=GraphResponse)
async def explore_graph(limit: int = 100):
    """Returns all entities and relationships for the graph explorer page."""
    try:
        nodes = get_all_entities(limit=limit)
        edges = get_all_relationships(limit=limit * 3)
        return GraphResponse(
            nodes=nodes,
            edges=edges,
            total_nodes=len(nodes),
            total_edges=len(edges),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/traverse", response_model=TraversalResponse)
async def traverse(request: TraversalRequest):
    """Multi-hop traversal from a named entity."""
    try:
        result = traverse_from_entity(
            start_entity=request.start_entity,
            max_depth=request.max_depth,
            relationship_types=request.relationship_types,
        )
        return TraversalResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
