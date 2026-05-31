from chunking_docs.vision.vlm_output import parse_vlm_output


def test_parse_vlm_output_from_json_object():
    parsed = parse_vlm_output(
        """
        ```json
        {
          "page_type": "map",
          "title": "River Corridor Diagram",
          "summary": "Shows connected station hubs.",
          "key_points": ["hub A", "hub B"],
          "visual_elements": ["legend"],
          "entities": ["river", "station"],
          "triples": [
            {"subject": "corridor", "predicate": "connects", "object": "station hub"}
          ]
        }
        ```
        """
    )

    assert parsed.caption == "River Corridor Diagram"
    assert "Shows connected station hubs." in parsed.summary
    assert "river" in parsed.summary
    assert "corridor connects station hub" in parsed.summary
    assert parsed.triples[0]["subject"] == "corridor"
    assert parsed.metadata["page_type"] == "map"
    assert parsed.metadata["vlm_parse_status"] == "json_object"


def test_parse_vlm_output_falls_back_to_raw_text():
    parsed = parse_vlm_output("plain visual summary")

    assert parsed.summary == "plain visual summary"
    assert parsed.triples == []
    assert parsed.metadata["vlm_parse_status"] == "raw_text"


def test_parse_vlm_output_repairs_truncated_json_object():
    parsed = parse_vlm_output(
        """
        ```json
        {
          "page_type": "text",
          "title": "Service Strategy",
          "summary": "Shows service improvement options.",
          "key_points": ["expand coverage", "improve transfer"],
          "visual_elements": [],
          "entities": [
            "transit",
            "service",
            "service",
            "transfer",
            "coverage",
            "network",
            "hub",
            "route",
            "extra"
        """
    )

    assert parsed.caption == "Service Strategy"
    assert "Shows service improvement options." in parsed.summary
    assert parsed.metadata["entities"] == [
        "transit",
        "service",
        "transfer",
        "coverage",
        "network",
        "hub",
        "route",
        "extra",
    ]
    assert parsed.metadata["vlm_parse_status"] == "json_repaired"
    assert parsed.metadata["vlm_parse_repaired"] is True


def test_parse_vlm_output_normalizes_object_underscore():
    parsed = parse_vlm_output(
        '[{"subject":"policy","predicate":"uses","object_":"river corridor"}]'
    )

    assert parsed.triples == [
        {"subject": "policy", "predicate": "uses", "object": "river corridor"}
    ]
