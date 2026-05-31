VISUAL_PROMPT_SCHEMA_VERSION = "visual_json_v3"


STRUCTURED_VISUAL_JSON_CONTRACT_KO = """
출력은 유효한 JSON 객체 하나만 작성하세요. 마크다운 코드블록, 설명문, 주석, JSON 밖의 문장을 쓰지 마세요.
반드시 다음 키를 포함하세요.
{
  "page_type": "text | table | chart | map | section_title | appendix | mixed | unknown",
  "title": "이미지에서 읽히는 가장 중요한 제목 또는 빈 문자열",
  "summary": "이미지가 전달하는 핵심 내용을 1~2문장으로 요약",
  "key_points": ["검색에 도움이 되는 핵심 주장, 수치, 라벨"],
  "visual_elements": ["표, 그래프, 지도, 범례, 축, 캡션, 색상/기호 의미"],
  "objects": [
    {
      "label": "눈에 보이는 객체 또는 영역 이름",
      "attributes": ["색상, 모양, 방향, 위치, 범례 의미 등"],
      "bbox": [0.0, 0.0, 1.0, 1.0],
      "confidence": 0.0
    }
  ],
  "entities": ["장소, 조직, 정책명, 지표, 연도, 시설명 등 명명된 항목"],
  "triples": [
    {
      "subject": "간결한 명사구",
      "predicate": "관계 동사 또는 관계명",
      "object": "간결한 명사구",
      "evidence": "이미지에서 읽히는 근거 문구 또는 시각 단서",
      "confidence": 0.0
    }
  ]
}
규칙:
- 이미지에서 직접 읽히거나 명확히 보이는 정보만 추출하세요.
- 불확실한 정보는 만들지 말고 배열을 비워 두세요.
- key_points는 최대 5개, visual_elements는 최대 6개, entities는 최대 8개로 제한하세요.
- objects는 객체 탐지나 시각 영역 설명에 도움이 될 때만 최대 8개 넣고, bbox를 모르면 키를 생략하세요.
- 같은 문자열을 배열 안에서 반복하지 마세요.
- 배열 제한에 도달하면 즉시 다음 키로 넘어가고 JSON 객체를 닫으세요.
- triples는 검색과 그래프 확장에 쓸 수 있는 관계 후보만 최대 3개 넣으세요.
- predicate는 관계를 설명하는 단어로 쓰고, `=`, `-`, `:` 같은 기호만 단독으로 쓰지 마세요.
- evidence는 30자 안팎의 짧은 근거 문구로 쓰세요.
- confidence는 0.0 이상 1.0 이하 숫자로 쓰세요.
- 모든 문자열은 한국어로 쓰되, 이미지에 영어 고유명이 있으면 원문을 유지하세요.
""".strip()


PAGE_SUMMARY_PROMPT_KO = f"""
이 이미지는 문서의 한 페이지입니다.
텍스트, 표, 차트, 지도, 제목, 캡션, 스캔 품질을 함께 보고 검색 가능한 근거를 추출하세요.

{STRUCTURED_VISUAL_JSON_CONTRACT_KO}
""".strip()


MAP_SUMMARY_PROMPT_KO = f"""
이 이미지는 지도, 공간 관계, 네트워크, 위치도, 구역도 또는 다이어그램일 수 있습니다.
지역/거점/축/권역/경계/자연 지형/교통축/범례 항목과 방향 관계를 우선 추출하세요.

{STRUCTURED_VISUAL_JSON_CONTRACT_KO}
""".strip()
