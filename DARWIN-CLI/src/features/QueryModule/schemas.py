"""
Darwin Query Module — JSON Schemas for Representation Outputs
Strict schemas used by the SkillRouter for input/output validation.
"""

from typing import Any, Dict


# ───────────────────────────────────────────────
# Comparison Table
# ───────────────────────────────────────────────
COMPARISON_TABLE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["headers", "rows"],
    "properties": {
        "headers": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["entity"],
                "properties": {
                    "entity": {"type": "string"},
                    "values": {"type": "array", "items": {"type": ["string", "number", "boolean", "null"]}},
                    "highlight": {"type": "boolean"},
                },
            },
        },
        "sortable": {"type": "boolean"},
        "filters": {"type": "array", "items": {"type": "string"}},
    },
}

# ───────────────────────────────────────────────
# Timeline
# ───────────────────────────────────────────────
TIMELINE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["events"],
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["date", "title"],
                "properties": {
                    "date": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "category": {"type": "string"},
                    "importance": {"type": "string", "enum": ["high", "medium", "low"]},
                },
            },
        },
        "zoom_levels": {"type": "array", "items": {"type": "string"}},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
}

# ───────────────────────────────────────────────
# Network Graph
# ───────────────────────────────────────────────
NETWORK_GRAPH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["nodes", "edges"],
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "type": {"type": "string"},
                    "metadata": {"type": "object"},
                },
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source", "target"],
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "label": {"type": "string"},
                    "type": {"type": "string"},
                    "weight": {"type": "number"},
                },
            },
        },
        "layout": {"type": "string", "enum": ["force", "hierarchical", "circular"]},
    },
}

# ───────────────────────────────────────────────
# Parameter Sandbox
# ───────────────────────────────────────────────
PARAMETER_SANDBOX_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["variables", "outcomes"],
    "properties": {
        "variables": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label", "type"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "type": {"type": "string", "enum": ["slider", "toggle", "select", "number"]},
                    "min": {"type": "number"},
                    "max": {"type": "number"},
                    "step": {"type": "number"},
                    "default": {"type": ["number", "boolean", "string"]},
                    "options": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "outcomes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "formula": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
        "scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "values": {"type": "object"},
                },
            },
        },
    },
}

# ───────────────────────────────────────────────
# Diagnostic Tree
# ───────────────────────────────────────────────
DIAGNOSTIC_TREE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["root"],
    "properties": {
        "root": {
            "type": "object",
            "required": ["id", "question"],
            "properties": {
                "id": {"type": "string"},
                "question": {"type": "string"},
                "description": {"type": "string"},
                "branches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["label", "next"],
                        "properties": {
                            "label": {"type": "string"},
                            "next": {"type": "string"},
                            "condition": {"type": "string"},
                        },
                    },
                },
            },
        },
        "nodes": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": ["question", "solution", "info"]},
                    "content": {"type": "string"},
                    "branches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "next": {"type": "string"},
                            },
                        },
                    },
                    "solution": {"type": "string"},
                    "severity": {"type": "string", "enum": ["critical", "warning", "info"]},
                },
            },
        },
    },
}

# ───────────────────────────────────────────────
# Stepper
# ───────────────────────────────────────────────
STEPPER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["steps"],
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["number", "title"],
                "properties": {
                    "number": {"type": "integer"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "code": {"type": "string"},
                    "checklist": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "current_step": {"type": "integer"},
    },
}

# ───────────────────────────────────────────────
# Accordion
# ───────────────────────────────────────────────
ACCORDION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["sections"],
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "content"],
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "expanded": {"type": "boolean"},
                },
            },
        },
    },
}

# ───────────────────────────────────────────────
# Summary Card
# ───────────────────────────────────────────────
SUMMARY_CARD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["title", "summary"],
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "metadata": {"type": "object"},
    },
}

# ───────────────────────────────────────────────
# Pros & Cons
# ───────────────────────────────────────────────
PROS_CONS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["subject", "pros", "cons"],
    "properties": {
        "subject": {"type": "string"},
        "pros": {"type": "array", "items": {"type": "string"}},
        "cons": {"type": "array", "items": {"type": "string"}},
        "verdict": {"type": "string"},
    },
}

# ───────────────────────────────────────────────
# Stats Grid
# ───────────────────────────────────────────────
STATS_GRID_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["stats"],
    "properties": {
        "stats": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "value"],
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": ["string", "number"]},
                    "unit": {"type": "string"},
                    "trend": {"type": "string", "enum": ["up", "down", "neutral"]},
                },
            },
        },
    },
}

# ───────────────────────────────────────────────
# Profile Card
# ───────────────────────────────────────────────
PROFILE_CARD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["entity"],
    "properties": {
        "entity": {"type": "string"},
        "attributes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "string"},
                },
            },
        },
        "description": {"type": "string"},
    },
}

# ───────────────────────────────────────────────
# Schema Registry
# ───────────────────────────────────────────────
SCHEMA_REGISTRY: Dict[str, Dict[str, Any]] = {
    "comparison_table": COMPARISON_TABLE_SCHEMA,
    "timeline": TIMELINE_SCHEMA,
    "network_graph": NETWORK_GRAPH_SCHEMA,
    "parameter_sandbox": PARAMETER_SANDBOX_SCHEMA,
    "diagnostic_tree": DIAGNOSTIC_TREE_SCHEMA,
    "stepper": STEPPER_SCHEMA,
    "accordion": ACCORDION_SCHEMA,
    "summary_card": SUMMARY_CARD_SCHEMA,
    "pros_cons": PROS_CONS_SCHEMA,
    "stats_grid": STATS_GRID_SCHEMA,
    "profile_card": PROFILE_CARD_SCHEMA,
    "markdown": {"type": "object", "properties": {"content": {"type": "string"}}},
    "relationship_map": NETWORK_GRAPH_SCHEMA,
}


def get_schema(representation: str) -> Dict[str, Any]:
    """Retrieve the JSON schema for a given representation type."""
    return SCHEMA_REGISTRY.get(representation, {"type": "object", "properties": {}})