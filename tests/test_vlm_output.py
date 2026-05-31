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
    assert parsed.triples[0]["subject"] == "corridor"
    assert parsed.metadata["page_type"] == "map"
    assert parsed.metadata["vlm_parse_status"] == "json_object"


def test_parse_vlm_output_falls_back_to_raw_text():
    parsed = parse_vlm_output("plain visual summary")

    assert parsed.summary == "plain visual summary"
    assert parsed.triples == []
    assert parsed.metadata["vlm_parse_status"] == "raw_text"


def test_parse_vlm_output_normalizes_object_underscore():
    parsed = parse_vlm_output(
        '[{"subject":"policy","predicate":"uses","object_":"river corridor"}]'
    )

    assert parsed.triples == [
        {"subject": "policy", "predicate": "uses", "object": "river corridor"}
    ]
