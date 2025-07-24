#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build an undirected drug–drug interaction graph in Neo4j.

The script expects a JSON file whose structure is:

{
    "DrugA": [
        {
            "int_drug_name": "DrugB",
            "int_drug_description": "description text"
        },
        ...
    ],
    ...
}

* Each pair (DrugA, DrugB) is treated as undirected; (A, B) and (B, A)
  are considered identical.
* Duplicate descriptions for the same pair are merged (newline-separated)
  to avoid redundancy.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship  # noqa: F401 (for type hints)


# --------------------------------------------------------------------------- #
# Data processing                                                             #
# --------------------------------------------------------------------------- #
def build_undirected_pairs(
    data: Dict[str, List[Dict[str, str]]]
) -> List[Dict[str, str]]:
    """
    Transform raw interaction data into an undirected, de-duplicated list.

    Returns
    -------
    List[Dict[str, str]]
        [
            {
                "source": "DrugA",
                "target": "DrugB",
                "description": "text1\\ntext2"
            },
            ...
        ]
    """
    edge_dict: Dict[tuple[str, str], List[str]] = defaultdict(list)

    for source, interactions in data.items():
        for interaction in interactions:
            target = interaction["int_drug_name"].strip()
            description = interaction["int_drug_description"].strip()

            # Sort the pair so (A, B) == (B, A).
            key = tuple(sorted((source, target)))

            if description not in edge_dict[key]:
                edge_dict[key].append(description)

    return [
        {
            "source": pair[0],
            "target": pair[1],
            "description": "\n".join(descs),
        }
        for pair, descs in edge_dict.items()
    ]


# --------------------------------------------------------------------------- #
# Neo4j writer                                                                #
# --------------------------------------------------------------------------- #
class Neo4jWriter:
    """Helper class for writing drug interaction data into Neo4j."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_constraints()

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        self._driver.close()

    def _create_constraints(self) -> None:
        """Ensure uniqueness on Drug nodes and on undirected relationships."""
        constraint_queries = [
            (
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (d:Drug) REQUIRE d.name IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR ()-[r:INTERACTS_WITH]-() "
                "REQUIRE (r.source, r.target) IS UNIQUE"
            ),
        ]

        with self._driver.session() as session:
            for query in constraint_queries:
                session.run(query)

    def write_pairs(self, pairs: List[Dict[str, str]]) -> None:
        """Bulk-insert drug pairs and merge duplicate descriptions."""
        cypher = """
        UNWIND $pairs AS p
        MERGE (a:Drug {name: p.source})
        MERGE (b:Drug {name: p.target})

        // Ensure a single direction per undirected edge
        WITH a, b, p,
             CASE WHEN p.source < p.target
                  THEN [p.source, p.target]
                  ELSE [p.target, p.source]
             END AS key

        MERGE (a)-[r:INTERACTS_WITH {source: key[0], target: key[1]}]->(b)
        ON CREATE SET r.description = p.description
        ON MATCH  SET r.description =
            CASE
                WHEN NOT p.description IN split(r.description, '\\n')
                THEN r.description + '\\n' + p.description
                ELSE r.description
            END
        """
        with self._driver.session() as session:
            session.write_transaction(lambda tx: tx.run(cypher, pairs=pairs))


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Convert drug interaction JSON into a Neo4j graph."
    )
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to Full_Drug_Interaction_Data.json",
    )
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", required=True)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    # ------------------- Load raw JSON ------------------- #
    with args.json.open(encoding="utf-8") as fp:
        raw_data: Dict[str, List[Dict[str, str]]] = json.load(fp)

    # ------------------- Build undirected pairs ---------- #
    drug_pairs = build_undirected_pairs(raw_data)
    print(f"Total undirected pairs: {len(drug_pairs)}")

    # ------------------- Write to Neo4j ------------------ #
    writer = Neo4jWriter(args.uri, args.user, args.password)
    writer.write_pairs(drug_pairs)
    writer.close()

    print("Neo4j import complete.")


if __name__ == "__main__":
    main()
