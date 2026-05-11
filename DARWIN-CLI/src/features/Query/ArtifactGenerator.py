# src/features/Query/ArtifactGenerator.py
"""
Generate the final structured artifact JSON using the smart LLM.
Now contains schemas for all supported ResponsePresentation types
(network_graph and code_walkthrough excluded).
"""
from __future__ import annotations

from typing import List
from src.features.LLM.LLMTypes import LLMResponse
from src.features.Query.QueryTypes import ResponsePresentation, PastedText, URLContent

# ═══════════════════════════════════════════════════════════════════════════
# JSON Schemas for every artifact type we support
# ═══════════════════════════════════════════════════════════════════════════

SCHEMA_STANDARD_DEFS = {
    "citation": {
        "type": "string",
        "description": "Source reference (e.g. 'book_of_enoch.pdf, page 5')"
    }
}

SCHEMA_REGISTRY: dict[ResponsePresentation, dict] = {

    # ── Simple content cards ───────────────────────────────────────────
    "summary_card": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "key_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "citation": {"$ref": "#/$defs/citation"}
                    },
                    "required": ["text"]
                }
            },
            "summary": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["title", "key_points", "summary"],
        "$defs": SCHEMA_STANDARD_DEFS
    },

    "profile_card": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "subtitle": {"type": "string"},
            "attributes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "value": {"type": "string"},
                        "citation": {"$ref": "#/$defs/citation"}
                    },
                    "required": ["label", "value"]
                }
            },
            "description": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["title", "attributes", "description"],
        "$defs": SCHEMA_STANDARD_DEFS
    },

    "stats_grid": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "value": {"type": "string"},
                        "citation": {"$ref": "#/$defs/citation"}
                    },
                    "required": ["label", "value"]
                }
            },
            "comparison": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "subjects": {"type": "array", "items": {"type": "string"}},
                        "values": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["label", "subjects", "values"]
                }
            }
        },
        "required": ["title", "metrics"],
        "$defs": SCHEMA_STANDARD_DEFS
    },

    "pros_cons": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "subject": {"type": "string"},
            "pros": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "citation": {"$ref": "#/$defs/citation"}
                    },
                    "required": ["text"]
                }
            },
            "cons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "citation": {"$ref": "#/$defs/citation"}
                    },
                    "required": ["text"]
                }
            },
            "verdict": {"type": "string"}
        },
        "required": ["title", "subject", "pros", "cons", "verdict"],
        "$defs": SCHEMA_STANDARD_DEFS
    },

    "accordion": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "body": {"type": "string"},
                        "citation_map": {
                            "type": "object",
                            "additionalProperties": {"type": "string"}
                        }
                    },
                    "required": ["heading", "body"]
                }
            }
        },
        "required": ["title", "sections"]
    },

    # ── Temporal / sequential ──────────────────────────────────────────
    "timeline": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date_or_period": {"type": "string"},
                        "description": {"type": "string"},
                        "citation": {"$ref": "#/$defs/citation"}
                    },
                    "required": ["date_or_period", "description"]
                }
            }
        },
        "required": ["title", "events"],
        "$defs": SCHEMA_STANDARD_DEFS
    },

    "stepper": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step_number": {"type": "integer"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "citation": {"$ref": "#/$defs/citation"}
                    },
                    "required": ["step_number", "title", "description"]
                }
            }
        },
        "required": ["title", "steps"],
        "$defs": SCHEMA_STANDARD_DEFS
    },

    # ── Relational / graphs ────────────────────────────────────────────
    "flowchart": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["id", "label"]
                }
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from_id": {"type": "string"},
                        "to_id": {"type": "string"},
                        "label": {"type": "string"}
                    },
                    "required": ["from_id", "to_id"]
                }
            },
            "layout_hint": {
                "type": "string",
                "enum": ["top-down", "left-right"]
            }
        },
        "required": ["title", "nodes", "edges"]
    },

    "relationship_map": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["id", "name"]
                }
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from_id": {"type": "string"},
                        "to_id": {"type": "string"},
                        "label": {"type": "string"},
                        "citation": {"$ref": "#/$defs/citation"}
                    },
                    "required": ["from_id", "to_id", "label"]
                }
            }
        },
        "required": ["title", "entities", "relationships"],
        "$defs": SCHEMA_STANDARD_DEFS
    },

    # ── Data ───────────────────────────────────────────────────────────
    "bar_chart": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "categories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "value": {"type": "number"}
                    },
                    "required": ["label", "value"]
                }
            },
            "x_axis_label": {"type": "string"},
            "y_axis_label": {"type": "string"}
        },
        "required": ["title", "categories"]
    },

    "comparison_table": {
        "type": "object",
        "properties": {
            "subjects": {"type": "array", "items": {"type": "string"}},
            "dimensions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dimension": {"type": "string"},
                        "values": {
                            "type": "object",
                            "additionalProperties": {"type": "string"}
                        }
                    },
                    "required": ["dimension", "values"]
                }
            }
        },
        "required": ["subjects", "dimensions"]
    },

    # ── Interactive ────────────────────────────────────────────────────
    "parameter_sandbox": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "parameters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "label": {"type": "string"},
                        "type": {"type": "string", "enum": ["slider", "toggle", "dropdown"]},
                        "min": {"type": "number"},
                        "max": {"type": "number"},
                        "step": {"type": "number"},
                        "default": {},
                        "options": {"type": "array", "items": {"type": "string"}},
                        "description": {"type": "string"}
                    },
                    "required": ["name", "label", "type", "default"]
                }
            },
            "recommended_experiments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "parameter_values": {
                            "type": "object",
                            "additionalProperties": {}
                        }
                    },
                    "required": ["label", "parameter_values"]
                }
            },
            "output_summary": {"type": "string"}
        },
        "required": ["title", "description", "parameters"]
    },

    "diagnostic_tree": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "tree": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "label": {"type": "string"},
                                "next": {
                                    "oneOf": [
                                        {"$ref": "#/$defs/tree_node"},
                                        {
                                            "type": "object",
                                            "properties": {
                                                "conclusion": {"type": "string"}
                                            },
                                            "required": ["conclusion"]
                                        }
                                    ]
                                }
                            },
                            "required": ["id", "label", "next"]
                        }
                    }
                },
                "required": ["question", "options"]
            },
            "total_questions": {"type": "integer"}
        },
        "required": ["title", "tree", "total_questions"],
        "$defs": {
            "tree_node": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "label": {"type": "string"},
                                "next": {
                                    "oneOf": [
                                        {"$ref": "#/$defs/tree_node"},
                                        {
                                            "type": "object",
                                            "properties": {
                                                "conclusion": {"type": "string"}
                                            },
                                            "required": ["conclusion"]
                                        }
                                    ]
                                }
                            },
                            "required": ["id", "label", "next"]
                        }
                    }
                },
                "required": ["question", "options"]
            }
        }
    },
}


SYSTEM_PROMPT = """You are a research assistant. You must answer using ONLY the provided context.
Never invent facts. If sources conflict, present both and note the contradiction.
Always cite the source filename and page for every factual claim.
Be neutral and objective – you mirror the provided material without adding opinions.
End with a brief summary of what the sources collectively say."""


async def generate_artifact(
    client,
    original_query: str,
    artifact_type: ResponsePresentation,
    resolved_contexts: List[str],
    blobs: List[PastedText] = (),
    url_contents: List[URLContent] = (),
    conversation_summary: str = "",
) -> LLMResponse:
    """Generate a structured artifact of the given type. Falls back to summary_card if no schema."""

    # If no schema for the chosen type, fallback to summary_card
    if artifact_type not in SCHEMA_REGISTRY:
        artifact_type = "summary_card"

    schema = SCHEMA_REGISTRY[artifact_type]

    # Build context block
    context_blocks = []
    for i, ctx in enumerate(resolved_contexts, 1):
        context_blocks.append(f"[Context {i}]:\n{ctx}")

    for b in blobs:
        context_blocks.append(f"[User-provided content]:\n{b.content}")

    for uc in url_contents:
        context_blocks.append(f"[External URL: {uc.url}]\n{uc.fetched_text}")

    full_context = "\n\n".join(context_blocks)

    system_msg = SYSTEM_PROMPT
    if conversation_summary:
        system_msg += f"\n\nPrevious conversation: {conversation_summary}"

    try:
        resp = await client.acall_smart(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Context:\n{full_context}\n\nUser query: {original_query}\n\nGenerate a {artifact_type} artifact."}
            ],
            json_schema=schema,
            temperature=0.1,
        )
        return resp
    except Exception as e:
        return LLMResponse(error=str(e))