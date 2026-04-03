"""Canonical concept registry with fuzzy deduplication via rapidfuzz.

Prevents concept explosion by detecting near-duplicate names/aliases
using token_sort_ratio scoring. English-only matching; Chinese names
are display-only.
"""

from __future__ import annotations

from rapidfuzz import fuzz, process

from knowledge.schemas import Concept


class ConceptRegistry:
    """Registry of canonical concepts with fuzzy dedup on add and resolve."""

    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}
        self._name_index: dict[str, str] = {}  # lowercased name/alias -> concept_id

    def add(self, concept: Concept) -> str | None:
        """Add a concept to the registry.

        Returns:
            None on success (new concept registered).
            Existing concept ID string if a near-duplicate or ID collision is found.
        """
        # Exact ID collision
        if concept.id in self._concepts:
            return concept.id

        # Fuzzy match concept name and aliases against all known names
        all_names = list(self._name_index.keys())
        if all_names:
            # Check the concept's primary name
            match = process.extractOne(
                concept.name.lower(),
                all_names,
                scorer=fuzz.token_sort_ratio,
            )
            if match is not None and match[1] >= 85:
                return self._name_index[match[0]]

            # Check each alias
            for alias in concept.aliases:
                match = process.extractOne(
                    alias.lower(),
                    all_names,
                    scorer=fuzz.token_sort_ratio,
                )
                if match is not None and match[1] >= 85:
                    return self._name_index[match[0]]

        # No duplicate found — register concept
        self._concepts[concept.id] = concept
        self._name_index[concept.name.lower()] = concept.id
        for alias in concept.aliases:
            self._name_index[alias.lower()] = concept.id

        return None

    def get(self, concept_id: str) -> Concept | None:
        """Retrieve a concept by its canonical ID."""
        return self._concepts.get(concept_id)

    def resolve(self, name: str, threshold: int = 70) -> str | None:
        """Resolve a fuzzy name/alias to its canonical concept ID.

        Args:
            name: The name or alias to look up.
            threshold: Minimum token_sort_ratio score (default 70).

        Returns:
            Canonical concept ID if match found above threshold, else None.
        """
        all_names = list(self._name_index.keys())
        if not all_names:
            return None
        match = process.extractOne(
            name.lower(),
            all_names,
            scorer=fuzz.token_sort_ratio,
        )
        if match is not None and match[1] >= threshold:
            return self._name_index[match[0]]
        return None

    def all_concepts(self) -> list[Concept]:
        """Return all registered concepts."""
        return list(self._concepts.values())

    def __len__(self) -> int:
        return len(self._concepts)
