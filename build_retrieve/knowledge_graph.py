"""
Knowledge graph construction and query module
- Build networkx knowledge graph from L2 node data
- BGE embedding similarity computation
- Graph retrieval (coarse ranking + fine ranking)
"""

import os
import pickle
import logging
from typing import Dict, List, Set, Tuple
from collections import defaultdict

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


def compute_text_similarity(
    queries: List[str],
    keys: List[str],
    embedding_model,
) -> np.ndarray:
    """
    Compute cosine similarity matrix between queries and keys.
    Returns a matrix of shape (len(queries), len(keys)).
    """
    if not queries or not keys:
        return np.array([])

    query_embs = embedding_model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    key_embs = embedding_model.encode(keys, convert_to_numpy=True, normalize_embeddings=True)

    # Cosine similarity = dot product of normalized vectors
    similarity = query_embs @ key_embs.T
    return similarity


def build_knowledge_graph(
    l2_entity_data: List[Dict],
    embedding_model,
    dedup_threshold: float = 0.7,
) -> Tuple[nx.DiGraph, Dict[str, Set[int]]]:
    """
    Build knowledge graph from L2 node entity/action/scene data.

    Args:
        l2_entity_data: List of L3NodeData, each containing:
            {
                "l2_index": int,
                "entities": [{"entity name": ..., "description": ...}],
                "actions": [{"entity name": ..., "action description": ...}],
                "scenes": [{"location": ...}],
            }
        embedding_model: SentenceTransformer model
        dedup_threshold: Entity deduplication similarity threshold

    Returns:
        (video_graph, entity_graph)
        video_graph: nx.DiGraph, nodes=L2 indices, edges=shared entities
        entity_graph: dict, entity_name -> set of L2 indices
    """
    video_graph = nx.DiGraph()
    entity_graph = defaultdict(set)

    for data in l2_entity_data:
        idx = data["l2_index"]

        # Parse entity/action/scene text
        entities = []
        for e in data.get("entities", []):
            name = e.get("entity name", "")
            desc = e.get("description", "")
            if name:
                entities.append(f"{name}, {desc}" if desc else name)

        actions = []
        for a in data.get("actions", []):
            name = a.get("entity name", "")
            action_desc = a.get("action description", "")
            if name:
                actions.append(f"{name}, {action_desc}" if action_desc else name)

        scenes = []
        for s in data.get("scenes", []):
            loc = s.get("location", "")
            if loc:
                scenes.append(loc)

        # Add graph node
        video_graph.add_node(
            idx,
            entities=entities,
            actions=actions,
            scenes=scenes,
        )

        # Process each entity text for deduplication and edge creation
        all_texts = entities + actions + scenes
        for text in all_texts:
            entity_name = text.split(",")[0].strip().lower()
            if not entity_name:
                continue

            if not entity_graph:
                # First entity, add directly
                entity_graph[entity_name].add(idx)
                continue

            # Compute similarity with existing entities
            existing_keys = list(entity_graph.keys())
            similarities = compute_text_similarity(
                [text], existing_keys, embedding_model
            )

            if similarities.size > 0:
                max_sim_idx = np.argmax(similarities[0])
                max_sim = similarities[0][max_sim_idx]

                if max_sim > dedup_threshold:
                    # Merge into existing entity
                    matched_key = existing_keys[max_sim_idx]
                    existing_nodes = entity_graph[matched_key]

                    # Add edges: current node with all other nodes sharing this entity
                    for other_idx in existing_nodes:
                        if other_idx != idx:
                            video_graph.add_edge(idx, other_idx, label=matched_key)
                            video_graph.add_edge(other_idx, idx, label=matched_key)

                    entity_graph[matched_key].add(idx)
                else:
                    # New entity
                    entity_graph[entity_name].add(idx)
            else:
                entity_graph[entity_name].add(idx)

    logger.info(
        f"Knowledge graph building complete: {video_graph.number_of_nodes()} nodes, "
        f"{video_graph.number_of_edges()} edges, "
        f"{len(entity_graph)} entities"
    )
    return video_graph, dict(entity_graph)


def retrieve_from_graph(
    query: str,
    video_graph: nx.DiGraph,
    entity_graph: Dict[str, Set[int]],
    embedding_model,
    reranker_model,
    top_k_embedding: int = 20,
    top_k_rerank: int = 5,
    similarity_threshold: float = 0.5,
) -> List[int]:
    """
    L3 knowledge graph retrieval: coarse ranking (BGE embedding) + fine ranking (BGE reranker).
    Returns a list of the most relevant L2 indices.
    """
    if not entity_graph or video_graph.number_of_nodes() == 0:
        logger.warning("Knowledge graph is empty, returning all nodes")
        return list(video_graph.nodes())

    candidate_indices = set()

    # --- Step 1: Coarse ranking via entity_graph ---
    entity_keys = list(entity_graph.keys())
    if entity_keys:
        similarities = compute_text_similarity(
            [query], entity_keys, embedding_model
        )
        if similarities.size > 0:
            for i, sim in enumerate(similarities[0]):
                if sim > similarity_threshold:
                    candidate_indices.update(entity_graph[entity_keys[i]])

    # --- Step 2: Supplement via video_graph node text ---
    for node_idx in video_graph.nodes():
        if node_idx in candidate_indices:
            continue
        node_data = video_graph.nodes[node_idx]
        node_texts = (
            node_data.get("entities", [])
            + node_data.get("actions", [])
            + node_data.get("scenes", [])
        )
        if not node_texts:
            continue
        sim_matrix = compute_text_similarity(
            [query], node_texts, embedding_model
        )
        if sim_matrix.size > 0 and np.mean(sim_matrix[0]) > similarity_threshold:
            candidate_indices.add(node_idx)

    candidate_indices = list(candidate_indices)
    if not candidate_indices:
        # No candidates matched, return all nodes
        logger.warning("No candidate nodes matched, returning all nodes")
        return list(video_graph.nodes())[:top_k_rerank]

    # Limit coarse ranking count
    if len(candidate_indices) > top_k_embedding:
        # Sort by embedding similarity and take top-k
        candidate_texts = []
        for idx in candidate_indices:
            node_data = video_graph.nodes[idx]
            text = " ".join(
                node_data.get("entities", [])
                + node_data.get("actions", [])
                + node_data.get("scenes", [])
            )
            candidate_texts.append(text)

        sim_scores = compute_text_similarity(
            [query], candidate_texts, embedding_model
        )[0]
        ranked = sorted(
            zip(candidate_indices, sim_scores), key=lambda x: -x[1]
        )
        candidate_indices = [idx for idx, _ in ranked[:top_k_embedding]]

    # --- Step 3: Fine ranking with cross-encoder ---
    if len(candidate_indices) <= top_k_rerank:
        return candidate_indices

    pairs = []
    for idx in candidate_indices:
        node_data = video_graph.nodes[idx]
        text = " ".join(
            node_data.get("entities", [])
            + node_data.get("actions", [])
            + node_data.get("scenes", [])
        )
        pairs.append((query, text))

    rerank_scores = reranker_model.predict(pairs)
    ranked = sorted(
        zip(candidate_indices, rerank_scores), key=lambda x: -x[1]
    )
    result = [idx for idx, _ in ranked[:top_k_rerank]]

    logger.info(f"L3 retrieval: {len(entity_graph)} entities -> {len(candidate_indices)} candidates -> {len(result)} results")
    return result


def save_graph(
    video_graph: nx.DiGraph,
    entity_graph: Dict[str, Set[int]],
    output_path: str,
):
    """Serialize knowledge graph to pickle"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        "video_graph": video_graph,
        "entity_graph": entity_graph,
    }
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Knowledge graph saved: {output_path}")


def load_graph(graph_path: str) -> Tuple[nx.DiGraph, Dict[str, Set[int]]]:
    """Load knowledge graph from pickle"""
    with open(graph_path, "rb") as f:
        data = pickle.load(f)
    return data["video_graph"], data["entity_graph"]
