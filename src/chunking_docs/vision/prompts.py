PAGE_SUMMARY_PROMPT_KO = """
이 이미지는 문서의 한 페이지입니다.
다음 항목을 JSON으로 정리하세요.
1. page_type: text, table, chart, map, section_title, appendix 중 하나
2. title: 페이지의 가장 중요한 제목
3. key_points: 핵심 주장 또는 수치 3~7개
4. visual_elements: 표, 그래프, 지도, 범례, 축, 캡션 설명
5. entities: 장소, 권역, 정책명, 지표, 연도
6. triples: subject, predicate, object 형태의 관계 후보
한국어로 답하세요. 확실하지 않은 내용은 추정이라고 표시하세요.
""".strip()


MAP_SUMMARY_PROMPT_KO = """
이 이미지는 지도 또는 공간 관계를 설명하는 도판일 수 있습니다.
제목, 지역/중심지/축/거점/자연 지형/교통축, 범례 항목, 이미지에서 읽히는 주요 방향을 정리하세요.
가능하면 관계를 triples 배열로 함께 추출하세요.
""".strip()
