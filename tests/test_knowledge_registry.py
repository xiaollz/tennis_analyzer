"""Tests for ConceptRegistry with fuzzy dedup via rapidfuzz."""

import pytest

from knowledge.schemas import Concept, ConceptType
from knowledge.registry import ConceptRegistry


def _make_concept(
    id: str,
    name: str,
    name_zh: str = "测试",
    aliases: list[str] | None = None,
    category: ConceptType = ConceptType.TECHNIQUE,
) -> Concept:
    return Concept(
        id=id,
        name=name,
        name_zh=name_zh,
        aliases=aliases or [],
        category=category,
        description=f"Test concept: {name}",
    )


@pytest.fixture
def registry() -> ConceptRegistry:
    return ConceptRegistry()


@pytest.fixture
def hip_rotation() -> Concept:
    return _make_concept(
        id="hip_rotation",
        name="Hip Rotation",
        name_zh="转髋",
        aliases=["hip turn", "hip drive"],
    )


@pytest.fixture
def shoulder_tilt() -> Concept:
    return _make_concept(
        id="shoulder_tilt",
        name="Shoulder Tilt",
        name_zh="肩倾斜",
    )


class TestRegistryAdd:
    def test_registry_add_success(self, registry: ConceptRegistry, hip_rotation: Concept):
        """Adding a new concept returns None (success), concept retrievable by ID."""
        result = registry.add(hip_rotation)
        assert result is None
        assert registry.get("hip_rotation") is not None
        assert registry.get("hip_rotation").name == "Hip Rotation"

    def test_registry_dedup(self, registry: ConceptRegistry, hip_rotation: Concept):
        """Adding 'Hip Turn' after 'Hip Rotation' exists returns 'hip_rotation' (near-duplicate)."""
        registry.add(hip_rotation)
        hip_turn_dup = _make_concept(id="hip_turn", name="Hip Turn", name_zh="转髋2")
        result = registry.add(hip_turn_dup)
        assert result == "hip_rotation"

    def test_registry_alias_dedup(self, registry: ConceptRegistry, hip_rotation: Concept):
        """Adding concept with alias matching existing concept's name triggers dedup."""
        registry.add(hip_rotation)
        # New concept whose alias 'Hip Rotation' matches existing name
        new_concept = _make_concept(
            id="hip_drive",
            name="Hip Drive Movement",
            name_zh="髋驱动",
            aliases=["Hip Rotation"],
        )
        result = registry.add(new_concept)
        assert result == "hip_rotation"

    def test_registry_no_false_positive(
        self, registry: ConceptRegistry, hip_rotation: Concept, shoulder_tilt: Concept
    ):
        """'Hip Rotation' and 'Shoulder Tilt' both added successfully (no false match)."""
        result1 = registry.add(hip_rotation)
        result2 = registry.add(shoulder_tilt)
        assert result1 is None
        assert result2 is None
        assert len(registry) == 2

    def test_registry_exact_id_collision(self, registry: ConceptRegistry, hip_rotation: Concept):
        """Adding concept with same ID as existing returns that ID."""
        registry.add(hip_rotation)
        same_id = _make_concept(id="hip_rotation", name="Completely Different Name", name_zh="不同")
        result = registry.add(same_id)
        assert result == "hip_rotation"


class TestRegistryResolve:
    def test_registry_resolve(self, registry: ConceptRegistry, hip_rotation: Concept):
        """resolve('hip drive') finds 'hip_rotation' when 'hip drive' is an alias."""
        registry.add(hip_rotation)
        result = registry.resolve("hip drive")
        assert result == "hip_rotation"

    def test_registry_resolve_no_match(self, registry: ConceptRegistry, hip_rotation: Concept):
        """resolve('completely unrelated term') returns None."""
        registry.add(hip_rotation)
        result = registry.resolve("completely unrelated term")
        assert result is None


class TestRegistryCount:
    def test_registry_count(self, registry: ConceptRegistry):
        """After adding 3 distinct concepts, len(registry) == 3."""
        concepts = [
            _make_concept(id="hip_rotation", name="Hip Rotation", name_zh="转髋"),
            _make_concept(id="shoulder_tilt", name="Shoulder Tilt", name_zh="肩倾斜"),
            _make_concept(id="unit_turn", name="Unit Turn", name_zh="整体转身"),
        ]
        for c in concepts:
            registry.add(c)
        assert len(registry) == 3

    def test_registry_all_concepts(self, registry: ConceptRegistry):
        """all_concepts() returns list of all registered concepts."""
        c1 = _make_concept(id="hip_rotation", name="Hip Rotation", name_zh="转髋")
        c2 = _make_concept(id="shoulder_tilt", name="Shoulder Tilt", name_zh="肩倾斜")
        registry.add(c1)
        registry.add(c2)
        all_c = registry.all_concepts()
        assert len(all_c) == 2
        ids = {c.id for c in all_c}
        assert ids == {"hip_rotation", "shoulder_tilt"}
