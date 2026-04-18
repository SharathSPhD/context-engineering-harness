AVACCHEDAKA_QUERY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema",
    "title": "AvacchedakaQuery",
    "type": "object",
    "required": ["qualificand", "condition"],
    "properties": {
        "qualificand": {"type": "string"},
        "qualifier": {"type": "string", "default": ""},
        "condition": {"type": "string"},
        "precision_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
        "max_elements": {"type": "integer", "default": 20},
    },
}

CONTEXT_ELEMENT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema",
    "title": "ContextElement",
    "type": "object",
    "required": ["id", "content", "precision", "avacchedaka"],
    "properties": {
        "id": {"type": "string"},
        "content": {"type": "string"},
        "precision": {"type": "number", "minimum": 0, "maximum": 1},
        "avacchedaka": {
            "type": "object",
            "required": ["qualificand", "qualifier", "condition"],
            "properties": {
                "qualificand": {"type": "string"},
                "qualifier": {"type": "string"},
                "condition": {"type": "string"},
                "relation": {"type": "string", "default": "inherence"},
            },
        },
        "sublated_by": {"type": ["string", "null"]},
        "provenance": {"type": "string"},
    },
}
