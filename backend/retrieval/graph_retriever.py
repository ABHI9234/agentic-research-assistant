from typing import List, Optional
from loguru import logger
from neo4j import GraphDatabase

from backend.app.config import get_settings

settings = get_settings()


def get_driver():
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )


def graph_search(
    query: str,
    top_k: int = None,
    depth: int = None,
) -> List[dict]:
    """
    Searches the Neo4j knowledge graph for entities matching the query,
    then traverses relationships up to `depth` hops to gather context.

    Returns chunks connected to matched entities.
    """
    if top_k is None:
        top_k = settings.retrieval_top_k
    if depth is None:
        depth = settings.graph_depth

    driver = get_driver()
    results = []

    try:
        with driver.session() as session:
            # Step 1: Find entities whose names appear in the query
            query_words = [
                w.strip() for w in query.split()
                if len(w.strip()) > 3
            ]

            if not query_words:
                return []

            # Build a case-insensitive entity match
            match_conditions = " OR ".join(
                [f"toLower(e.name) CONTAINS toLower('{w}')" for w in query_words[:5]]
            )

            cypher = f"""
                MATCH (e:Entity)
                WHERE {match_conditions}
                WITH e LIMIT 10
                MATCH path = (e)-[*1..{depth}]-(related:Entity)
                WITH e, related, path
                MATCH (related)-[:APPEARS_IN]->(c:Chunk)-[:BELONGS_TO]->(d:Document)
                RETURN DISTINCT
                    c.chunk_id AS chunk_id,
                    c.text AS text,
                    c.page_num AS page_num,
                    d.filename AS filename,
                    d.doc_id AS doc_id,
                    e.name AS matched_entity,
                    length(path) AS hop_distance
                LIMIT {top_k * 2}
            """

            records = session.run(cypher)
            seen_chunks = set()

            for record in records:
                chunk_id = record.get("chunk_id")
                if chunk_id in seen_chunks:
                    continue
                seen_chunks.add(chunk_id)

                hop = record.get("hop_distance", 1)
                # Score: closer hops score higher
                score = round(1.0 / (1.0 + hop * 0.3), 4)

                results.append(
                    {
                        "chunk_id": chunk_id or "",
                        "text": record.get("text") or "",
                        "source_file": record.get("filename") or "",
                        "page_num": record.get("page_num") or 0,
                        "doc_id": record.get("doc_id") or "",
                        "relevance_score": score,
                        "retrieval_method": "graph",
                        "matched_entity": record.get("matched_entity") or "",
                        "hop_distance": hop,
                    }
                )

        logger.debug(f"Graph search: '{query[:60]}' → {len(results)} results")
        return results[:top_k]

    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        return []
    finally:
        driver.close()


def get_all_entities(limit: int = 200) -> List[dict]:
    """Returns all entity nodes — used by the Knowledge Graph Explorer page."""
    driver = get_driver()
    try:
        with driver.session() as session:
            records = session.run(
                """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[:APPEARS_IN]->(c:Chunk)-[:BELONGS_TO]->(d:Document)
                RETURN e.name AS name,
                       e.type AS type,
                       e.description AS description,
                       collect(DISTINCT d.doc_id) AS doc_ids,
                       count(c) AS mention_count
                ORDER BY mention_count DESC
                LIMIT $limit
                """,
                limit=limit,
            )
            return [
                {
                    "entity_id": r["name"],
                    "name": r["name"],
                    "type": r["type"] or "CONCEPT",
                    "description": r["description"] or "",
                    "doc_ids": list(r["doc_ids"]),
                    "mention_count": r["mention_count"],
                }
                for r in records
            ]
    except Exception as e:
        logger.error(f"get_all_entities failed: {e}")
        return []
    finally:
        driver.close()


def get_all_relationships(limit: int = 500) -> List[dict]:
    """Returns all relationships — used by the Knowledge Graph Explorer page."""
    driver = get_driver()
    try:
        with driver.session() as session:
            records = session.run(
                """
                MATCH (a:Entity)-[r]->(b:Entity)
                RETURN a.name AS source,
                       b.name AS target,
                       type(r) AS rel_type
                LIMIT $limit
                """,
                limit=limit,
            )
            return [
                {
                    "source_entity": r["source"],
                    "target_entity": r["target"],
                    "relationship_type": r["rel_type"],
                    "weight": 1.0,
                }
                for r in records
            ]
    except Exception as e:
        logger.error(f"get_all_relationships failed: {e}")
        return []
    finally:
        driver.close()


def traverse_from_entity(
    start_entity: str,
    max_depth: int = 2,
    relationship_types: Optional[List[str]] = None,
) -> dict:
    """Multi-hop traversal starting from a named entity."""
    driver = get_driver()
    try:
        with driver.session() as session:
            rel_filter = ""
            if relationship_types:
                types = "|".join(relationship_types)
                rel_filter = f":{types}"

            cypher = f"""
                MATCH path = (start:Entity {{name: $name}})-[{rel_filter}*1..{max_depth}]-(other:Entity)
                RETURN DISTINCT
                    other.name AS name,
                    other.type AS type,
                    other.description AS description,
                    length(path) AS depth,
                    [n IN nodes(path) | n.name] AS path_nodes
                LIMIT 100
            """
            records = session.run(cypher, name=start_entity)

            nodes = []
            paths = []
            seen = set()

            for r in records:
                name = r["name"]
                if name not in seen:
                    seen.add(name)
                    nodes.append(
                        {
                            "entity_id": name,
                            "name": name,
                            "type": r["type"] or "CONCEPT",
                            "description": r["description"] or "",
                            "doc_ids": [],
                            "mention_count": 0,
                        }
                    )
                path = r["path_nodes"]
                if path:
                    paths.append([str(n) for n in path])

            return {
                "start_entity": start_entity,
                "depth_reached": max_depth,
                "nodes": nodes,
                "edges": [],
                "paths": paths,
            }
    except Exception as e:
        logger.error(f"Traversal failed: {e}")
        return {"start_entity": start_entity, "depth_reached": 0, "nodes": [], "edges": [], "paths": []}
    finally:
        driver.close()
