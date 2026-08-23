from typing import Generator
from loguru import logger
from groq import Groq
from neo4j import GraphDatabase
import json
import re

from backend.app.config import get_settings

settings = get_settings()


def get_neo4j_driver():
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )


def get_groq_client():
    return Groq(api_key=settings.groq_api_key)


def extract_entities_from_text(text: str) -> dict:
    client = get_groq_client()

    prompt = f"""Extract entities from this text. Return ONLY a JSON object. No explanation. No markdown. No code blocks. No thinking.

Text: {text[:600]}

Return exactly this format:
{{"entities":[{{"name":"X","type":"CONCEPT","description":"brief"}}],"relationships":[{{"source":"X","target":"Y","type":"RELATED_TO"}}]}}

JSON:"""

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()

        # Strip thinking tags if present
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

        # Strip markdown code blocks
        raw = re.sub(r'```json|```', '', raw).strip()

        # Find JSON object in response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)

        parsed = json.loads(raw)
        return parsed

    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        return {"entities": [], "relationships": []}


def store_in_neo4j(doc_id, filename, chunk_id, chunk_text, page_num, extracted, driver):
    entities = extracted.get("entities", [])
    relationships = extracted.get("relationships", [])

    with driver.session() as session:
        session.run(
            "MERGE (d:Document {doc_id: $doc_id}) ON CREATE SET d.filename = $filename",
            doc_id=doc_id, filename=filename,
        )
        session.run(
            """MERGE (c:Chunk {chunk_id: $chunk_id})
               ON CREATE SET c.text = $text, c.page_num = $page_num
               WITH c MATCH (d:Document {doc_id: $doc_id})
               MERGE (c)-[:BELONGS_TO]->(d)""",
            chunk_id=chunk_id, text=chunk_text[:300], page_num=page_num, doc_id=doc_id,
        )
        for entity in entities:
            name = entity.get("name", "").strip()
            if not name:
                continue
            session.run(
                """MERGE (e:Entity {name: $name})
                   ON CREATE SET e.type = $type, e.description = $description
                   WITH e MATCH (c:Chunk {chunk_id: $chunk_id})
                   MERGE (e)-[:APPEARS_IN]->(c)""",
                name=name, type=entity.get("type", "CONCEPT"),
                description=entity.get("description", ""), chunk_id=chunk_id,
            )
        for rel in relationships:
            source = rel.get("source", "").strip()
            target = rel.get("target", "").strip()
            rel_type = rel.get("type", "RELATED_TO").upper().replace(" ", "_")
            if not source or not target:
                continue
            try:
                session.run(
                    f"MERGE (a:Entity {{name:$s}}) MERGE (b:Entity {{name:$t}}) MERGE (a)-[:{rel_type}]->(b)",
                    s=source, t=target,
                )
            except Exception as e:
                logger.warning(f"Rel store failed: {e}")

    return len(entities), len(relationships)


async def extract_and_store_entities(filepath: str, job_id: str) -> dict:
    from backend.ingestion.chunker import stream_chunks

    driver = get_neo4j_driver()
    total_entities = 0
    total_relationships = 0
    chunks_processed = 0
    MAX_CHUNKS = 10

    try:
        for chunk in stream_chunks(filepath):
            if chunks_processed >= MAX_CHUNKS:
                break
            extracted = extract_entities_from_text(chunk["text"])
            ents, rels = store_in_neo4j(
                doc_id=chunk["doc_id"],
                filename=chunk["filename"],
                chunk_id=chunk["chunk_id"],
                chunk_text=chunk["text"],
                page_num=chunk["page_num"],
                extracted=extracted,
                driver=driver,
            )
            total_entities += ents
            total_relationships += rels
            chunks_processed += 1
    finally:
        driver.close()

    logger.info(f"[{job_id}] Entities: {total_entities}, Relationships: {total_relationships}")
    return {"entities_stored": total_entities, "relationships_stored": total_relationships}
