# chunking_docs

도시계획 PDF를 RAG용 데이터로 만들기 위한 청킹·임베딩 라이브러리다.

현재 목표 문서는 서울시 `2030 서울도시기본계획` PDF다.

원본: https://urban.seoul.go.kr/UpisArchive/DATA/PWEB/STATIC/1_seoul_plan.pdf

## 현재 범위

- PDF 다운로드
- 페이지 profile 생성
- 텍스트 레이어 품질 판정
- 목차 기반 section metadata 부여
- page-level starter chunk 생성
- BM25 lexical index
- Qdrant upsert record와 local mode 검증
- PostgreSQL 적재 row/schema/optional writer
- OCR/VLM/graph extraction interface
- local dense+BM25+graph hybrid search

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Qdrant 연동까지 쓰려면:

```bash
pip install -e ".[qdrant,dev]"
```

로컬 GPU에서 dense/VLM 실험까지 하려면:

```bash
pip install -e ".[embeddings,vision,qdrant,dev]"
```

## 사용 예시

```bash
chunking-docs download \
  "https://urban.seoul.go.kr/UpisArchive/DATA/PWEB/STATIC/1_seoul_plan.pdf" \
  data/raw/1_seoul_plan.pdf

chunking-docs profile data/raw/1_seoul_plan.pdf --output-dir outputs/profile
chunking-docs render data/raw/1_seoul_plan.pdf --pages 1,4,16,30,100,150,188
chunking-docs chunk data/raw/1_seoul_plan.pdf --output outputs/chunks.jsonl
chunking-docs package data/raw/1_seoul_plan.pdf --output-dir outputs/package
```

`package` 명령은 다음 파일을 만든다.

- `pages.jsonl`: 페이지 profile
- `chunks.jsonl`: 청킹 결과
- `assets.jsonl`: 페이지/지도/표/그림 asset manifest
- `triples.jsonl`: graph triple 후보
- `bm25_tokens.json`: lexical search token manifest
- `qdrant_text_records.jsonl`: Qdrant upsert용 dry-run text vector record
- `qdrant_image_records.jsonl`: Qdrant upsert용 dry-run image vector record
- `qdrant_caption_records.jsonl`: Qdrant upsert용 dry-run caption vector record
- `qdrant_collection.json`: Qdrant collection 설계 manifest

OCR/VLM 주석을 붙일 때:

```bash
chunking-docs annotate-assets --package-dir outputs/package --ocr tesseract --in-place
chunking-docs annotate-assets \
  --package-dir outputs/package \
  --vlm hf \
  --vlm-model <local-or-huggingface-vlm-model> \
  --pages 100,150,188 \
  --in-place
```

`--in-place`와 기본 `--rebuild-search`를 함께 쓰면 OCR/VLM 결과가 `chunks.jsonl`, `bm25_tokens.json`, `qdrant_text_records.jsonl`에 반영된다.

외부 VLM 또는 사람이 검수한 주석을 JSONL로 반영할 수도 있다.

```bash
chunking-docs apply-annotations examples/seoul_plan_seed_annotations.jsonl --package-dir outputs/package --in-place
chunking-docs split-chunks --package-dir outputs/package --max-chars 600 --overlap-chars 80 --in-place
chunking-docs search-local "동북권 발전구상 중랑천" --package-dir outputs/package --top-k 5
chunking-docs search-local "동북권 발전구상 중랑천" --package-dir outputs/package --graph-expand --top-k 5
chunking-docs export-graph --package-dir outputs/package
chunking-docs audit-package --package-dir outputs/package
chunking-docs eval-retrieval examples/seoul_plan_retrieval_cases.jsonl --package-dir outputs/package --top-k 5
```

`apply-annotations`는 `assets.jsonl`, `chunks.jsonl`, `triples.jsonl`, BM25, Qdrant dry-run records를 갱신한다.
`audit-package`는 누락 chunk, orphan triple, 남은 OCR/VLM 대상 페이지를 확인한다.
`eval-retrieval`은 seed query가 기대 페이지를 top-k 안에서 찾는지 검증한다.

## Qdrant 로컬 실행

```bash
docker compose -f docker-compose.qdrant.yml up -d
chunking-docs qdrant-upsert --records outputs/package/qdrant_text_records.jsonl
chunking-docs qdrant-upsert-package --package-dir outputs/package
```

Docker가 없는 환경에서는 qdrant-client local mode로 upsert 로직을 검증할 수 있다.

```bash
chunking-docs qdrant-upsert-package --package-dir outputs/package --location ':memory:'
```

## PostgreSQL 적재

PostgreSQL은 원본 문서/페이지/chunk/asset/triple metadata 보관용이다. vector 검색은 기본적으로 Qdrant가 담당한다.

```bash
pip install -e ".[postgres]"
chunking-docs postgres-rows --package-dir outputs/package
chunking-docs postgres-upsert "postgresql://user:password@localhost:5432/chunking_docs" --package-dir outputs/package
```

## 설계 문서

- [문서 관찰 메모](docs/seoul_plan_observations.md)
- [아키텍처](docs/architecture.md)

## 저장소 상태

`JMTdongsan/chunking_docs` 원격 저장소가 아직 존재하지 않거나 현재 환경에서 생성 권한을 확인할 수 없다.
이 디렉터리는 독립 로컬 Git 저장소로 초기화해 사용할 수 있다.
