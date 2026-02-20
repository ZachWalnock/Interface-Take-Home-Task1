from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

from .config import Settings
from .schemas import Entity, Relation
from .utils import make_uuid, utc_now_iso


@dataclass(slots=True)
class Neo4jGraphStore:
    uri: str
    user: str
    password: str
    database: str
    allow_db_fallback: bool = True
    fallback_database: str = "neo4j"
    _driver: Any = field(init=False, repr=False)
    active_database: str = field(init=False)

    def __post_init__(self) -> None:
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.active_database = self._resolve_active_database()

    @classmethod
    def from_settings(cls, settings: Settings) -> "Neo4jGraphStore":
        return cls(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
            allow_db_fallback=settings.neo4j_allow_db_fallback,
            fallback_database=settings.neo4j_fallback_database,
        )

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jGraphStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def create_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT measurement_id_unique IF NOT EXISTS FOR (m:Measurement) REQUIRE m.measurement_id IS UNIQUE",
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT page_key IF NOT EXISTS FOR (p:Page) REQUIRE (p.doc_id, p.page_number) IS UNIQUE",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_ref_index IF NOT EXISTS FOR (e:Entity) ON (e.normalized_ref)",
        ]
        with self._session(database=self.active_database) as session:
            for cypher in statements:
                session.run(cypher)

    def upsert_document(self, doc_id: str, source_path: str | Path, page_count: int) -> None:
        source_path = str(source_path)
        cypher = """
        MERGE (d:Document {doc_id: $doc_id})
        ON CREATE SET d.created_at = $now
        SET d.source_path = $source_path,
            d.page_count = $page_count,
            d.updated_at = $now
        """
        with self._session(database=self.active_database) as session:
            session.run(
                cypher,
                doc_id=doc_id,
                source_path=source_path,
                page_count=page_count,
                now=utc_now_iso(),
            )

    def upsert_entity(self, doc_id: str, source_path: str | Path, entity: Entity) -> None:
        if not entity.entity_id:
            raise ValueError("entity.entity_id is required before writing to Neo4j.")

        source_path = str(source_path)
        bbox = entity.bbox.model_dump() if entity.bbox else None
        entity_props = {
            "entity_id": entity.entity_id,
            "name": entity.name,
            "entity_type": entity.entity_type,
            "description": entity.description,
            "page_number": entity.page_number,
            "aliases": entity.aliases,
            "confidence": entity.confidence,
            "doc_id": doc_id,
            "normalized_ref": entity.name.upper().replace("-", "").replace("_", "").replace(" ", ""),
            "attributes_json": _to_json_string(entity.attributes),
            "bbox_left": bbox.get("left") if bbox else None,
            "bbox_top": bbox.get("top") if bbox else None,
            "bbox_right": bbox.get("right") if bbox else None,
            "bbox_bottom": bbox.get("bottom") if bbox else None,
            "bbox_label": bbox.get("label") if bbox else None,
        }
        entity_props = _sanitize_property_map(entity_props)
        measurement_rows: list[dict[str, Any]] = []
        for measurement in entity.measurements:
            mbbox = measurement.bbox.model_dump() if measurement.bbox else None
            row = {
                "measurement_id": make_uuid("msr"),
                "kind": measurement.kind,
                "value": measurement.value,
                "unit": measurement.unit,
                "raw_text": measurement.raw_text,
                "confidence": measurement.confidence,
                "bbox_left": mbbox.get("left") if mbbox else None,
                "bbox_top": mbbox.get("top") if mbbox else None,
                "bbox_right": mbbox.get("right") if mbbox else None,
                "bbox_bottom": mbbox.get("bottom") if mbbox else None,
                "bbox_label": mbbox.get("label") if mbbox else None,
                "page_number": entity.page_number,
                "doc_id": doc_id,
            }
            measurement_rows.append(_sanitize_property_map(row))

        cypher = """
        MERGE (d:Document {doc_id: $doc_id})
        ON CREATE SET d.created_at = $now
        SET d.source_path = $source_path, d.updated_at = $now
        MERGE (p:Page {doc_id: $doc_id, page_number: $page_number})
        ON CREATE SET p.created_at = $now
        SET p.updated_at = $now
        MERGE (d)-[:HAS_PAGE]->(p)
        MERGE (e:Entity {entity_id: $entity_id})
        ON CREATE SET e.created_at = $now
        SET e += $entity_props, e.updated_at = $now
        MERGE (p)-[:MENTIONS]->(e)
        WITH e
        UNWIND $measurements AS m
        MERGE (me:Measurement {measurement_id: m.measurement_id})
        ON CREATE SET me.created_at = $now
        SET me += m, me.updated_at = $now
        MERGE (e)-[:HAS_MEASUREMENT]->(me)
        """
        with self._session(database=self.active_database) as session:
            session.run(
                cypher,
                now=utc_now_iso(),
                doc_id=doc_id,
                source_path=source_path,
                page_number=entity.page_number,
                entity_id=entity.entity_id,
                entity_props=entity_props,
                measurements=measurement_rows,
            )

    def upsert_relation(self, relation: Relation, source_entity_id: str, target_entity_id: str) -> None:
        relation_id = relation.relation_id or make_uuid("rel")
        rel_props = {
            "relation_id": relation_id,
            "relation_type": relation.relation_type,
            "line_id": relation.line_id,
            "description": relation.description,
            "page_number": relation.page_number,
            "attributes_json": _to_json_string(relation.attributes),
            "confidence": relation.confidence,
            "source_ref": relation.source_ref,
            "target_ref": relation.target_ref,
        }
        rel_props = _sanitize_property_map(rel_props)
        cypher = """
        MATCH (s:Entity {entity_id: $source_entity_id})
        MATCH (t:Entity {entity_id: $target_entity_id})
        MERGE (s)-[r:CONNECTED_TO {relation_id: $relation_id}]->(t)
        SET r += $rel_props
        """
        with self._session(database=self.active_database) as session:
            session.run(
                cypher,
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                relation_id=relation_id,
                rel_props=rel_props,
            )

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._session(database=self.active_database) as session:
            result = session.run(cypher, params or {})
            return [row.data() for row in result]

    def _resolve_active_database(self) -> str:
        configured = self.database
        if self._database_exists(configured):
            return configured

        if self.allow_db_fallback and self.fallback_database and self.fallback_database != configured:
            if self._database_exists(self.fallback_database):
                print(
                    f"[neo4j] Warning: configured database '{configured}' was not found at {self.uri}. "
                    f"Falling back to '{self.fallback_database}'.",
                    flush=True,
                )
                return self.fallback_database

        raise RuntimeError(
            f"Neo4j database '{configured}' does not exist at {self.uri}. "
            "Update NEO4J_DATABASE or point NEO4J_URI to the correct instance."
        )

    def _database_exists(self, database_name: str) -> bool:
        try:
            with self._session(database=database_name) as session:
                session.run("RETURN 1 AS ok").single()
            return True
        except ClientError as exc:
            if _is_database_not_found(exc):
                return False
            raise

    def _session(self, *, database: str):
        # Disable notification warnings from Neo4j for cleaner terminal output.
        return self._driver.session(database=database, notifications_min_severity="OFF")


def _is_database_not_found(exc: ClientError) -> bool:
    code = (exc.code or "").strip()
    text = str(exc)
    return code.endswith("Database.DatabaseNotFound") or "Database does not exist" in text


def _to_json_string(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return "{}"


def _sanitize_property_map(props: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in props.items():
        sanitized[key] = _sanitize_property_value(value)
    return sanitized


def _sanitize_property_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            sanitized_item = _sanitize_property_value(item)
            if sanitized_item is None or isinstance(sanitized_item, (str, bool, int, float)):
                out.append(sanitized_item)
            else:
                out.append(json.dumps(sanitized_item, ensure_ascii=True, separators=(",", ":")))
        return out
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return str(value)
