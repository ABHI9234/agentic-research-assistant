import asyncio
from typing import List
from loguru import logger
from groq import Groq
from neo4j import GraphDatabase
import json
import re

from backend.app.config import get_settings

settings = get_settings()

BATCH_SIZE = 4
CONCURRENCY = 2
MAX_CHUNKS = 100


def get_neo4j_driver():
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )


def get_groq_client():
    return Groq(api_key=settings.groq_api_key)


def extract_entities_from_batch(chunks: List[dict]) -> dict:
    client = get_groq_client()

    labeled_text = "\n\n".join(
        f"[Chunk {i+1}]\n{c['text'][:500]}" for i, c in enumerate(chunks)
    )

    prompt = f"""You are an expert knowledge graph builder for enterprise technology and HR documents.

Extract ALL entities and relationships from the labeled chunks below.
Entity types to use: ORGANIZATION, PRODUCT, TECHNOLOGY, PERSON, CONCEPT, LOCATION, ROLE, SKILL
Relationship types to use: DEVELOPS, ACQUIRES, USES, PART_OF, COMPETES_WITH, LEADS, RELATED_TO, POWERS, PROVIDES, REQUIRES

Each entity MUST include "chunk_index" — the number of the [Chunk N] label it was found in.

{labeled_text}

Return ONLY a raw JSON object. No markdown. No backticks. No explanation.
Use this exact structure:
{{"entities":[{{"name":"Cisco Systems","type":"ORGANIZATION","description":"Networking hardware and software company","chunk_index":1}}],"relationships":[{{"source":"Cisco Systems","target":"IOS","type":"DEVELOPS"}}]}}"""

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
            reasoning_effort="low",
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        raw = re.sub(r'```json|```', '', raw).strip()
        raw = re.sub(r',\s*([}\]])', r'\1', raw)

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)

        parsed = json.loads(raw)
        if "entities" not in parsed:
            parsed["entities"] = []
        if "relationships" not in parsed:
            parsed["relationships"] = []
        return parsed

    except Exception as e:
        logger.exception(f"Batch entity extraction failed: {e}")
        return {"entities": [], "relationships": []}


def ensure_chunk_node(doc_id, filename, chunk_id, chunk_text, page_num, driver):
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


def store_entities_for_chunk(entities, chunk_id, driver):
    with driver.session() as session:
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


def store_relationships(relationships, driver):
    with driver.session() as session:
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


async def extract_and_store_entities(filepath: str, job_id: str) -> dict:
    from backend.ingestion.chunker import stream_chunks

    driver = get_neo4j_driver()
    total_entities = 0
    total_relationships = 0

    chunks = []
    for chunk in stream_chunks(filepath):
        if len(chunks) >= MAX_CHUNKS:
            break
        chunks.append(chunk)

    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
    logger.info(f"[{job_id}] {len(chunks)} chunks -> {len(batches)} batches, concurrency={CONCURRENCY}")

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def process_batch(batch):
        async with semaphore:
            extracted = await asyncio.to_thread(extract_entities_from_batch, batch)
            return batch, extracted

    results = await asyncio.gather(*[process_batch(b) for b in batches])

    try:
        for batch, _ in results:
            for chunk in batch:
                ensure_chunk_node(
                    doc_id=chunk["doc_id"], filename=chunk["filename"],
                    chunk_id=chunk["chunk_id"], chunk_text=chunk["text"],
                    page_num=chunk["page_num"], driver=driver,
                )

        for batch, extracted in results:
            entities = extracted.get("entities", [])
            relationships = extracted.get("relationships", [])

            by_chunk_index = {}
            for e in entities:
                idx = e.get("chunk_index", 1)
                by_chunk_index.setdefault(idx, []).append(e)

            for idx, batch_entities in by_chunk_index.items():
                pos = idx - 1
                if 0 <= pos < len(batch):
                    chunk_id = batch[pos]["chunk_id"]
                    store_entities_for_chunk(batch_entities, chunk_id, driver)
                    total_entities += len(batch_entities)

            store_relationships(relationships, driver)
            total_relationships += len(relationships)
    finally:
        driver.close()

    logger.info(f"[{job_id}] Entities: {total_entities}, Relationships: {total_relationships}")
    return {"entities_stored": total_entities, "relationships_stored": total_relationships}
