"""
Subsystem configuration file.

This file defines all available subsystems for routing classification.
Add new subsystems here as needed.
"""

# List of all subsystem identifiers
SUBSYSTEMS = [
    "material_knowledge",
    "causal_regulation_lookup",
    "pure_regulation_lookup",
    "acquire_image_example",
]

# Optional: Subsystem metadata (descriptions, etc.)
SUBSYSTEM_METADATA = {
    "material_knowledge": {
        "name": "Material Knowledge",
        "description": "Handles queries about material properties, differences, and characteristics"
    },
    "causal_regulation_lookup": {
        "name": "Causal Regulation Lookup",
        "description": "Handles queries about regulations, standards, and compliance requirements with specific context (e.g., design phase, construction phase, specific scenarios)"
    },
    "pure_regulation_lookup": {
        "name": "Pure Regulation Lookup",
        "description": "Handles queries about regulations, standards, and compliance requirements without specific contextual information"
    },
    "acquire_image_example": {
        "name": "Acquire Image Example",
        "description": "Handles queries that require image acquisition or examples"
    }
}

def get_all_subsystems():
    """Return list of all subsystem identifiers."""
    return SUBSYSTEMS.copy()

def get_subsystem_metadata(subsystem_id):
    """Return metadata for a specific subsystem."""
    return SUBSYSTEM_METADATA.get(subsystem_id, {})

