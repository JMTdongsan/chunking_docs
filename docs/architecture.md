# Chunking Docs Architecture

## 목표

`chunking_docs`는 도시계획 PDF를 RAG에 넣기 전 단계의 전처리 라이브러리다. 산출물은 단일 텍스트 chunk가 아니라 다음을 모두 포함한다.

- 텍스트 chunk
- 표/그래프/지도/그림 asset
- OCR 텍스트
- VLM 요약
- dense embedding
- BM25 lexical index
- graph triple
- Qdrant 적재 record
- 추후 PostgreSQL 적재 가능한 정규화 record

## 파이프라인

1. **Document intake**
   - 원본 PDF 다운로드
   - 파일 hash 기반 `doc_id` 생성
   - PDF metadata와 페이지 수 저장

2. **Page profiling**
   - 페이지 크기, 텍스트 길이, 이미지 개수, drawing 개수 측정
   - 텍스트 레이어 품질을 `good`, `degraded`, `empty`로 분류
   - 시각 정보가 많은 페이지를 별도 큐로 보냄

3. **Layout and section mapping**
   - 목차에서 장/절 범위를 구성
   - 각 chunk에 `chapter`, `section`, `issue` 메타데이터 부여
   - 향후 OCR 기반 제목 검출로 section map 자동화

4. **Text/OCR/VLM extraction**
   - 텍스트 레이어가 좋은 페이지: PDF text를 정제
   - 텍스트 레이어가 깨진 페이지: OCR을 실행
   - 지도·표·그래프·장 전환 페이지: VLM summary 생성
   - RTX 5090 환경에서는 VLM backend를 `transformers` 또는 별도 inference server로 연결

5. **Chunk generation**
   - page text chunk
   - table chunk
   - chart chunk
   - map chunk
   - page summary chunk
   - section summary chunk

6. **Embedding generation**
   - `text_dense`: 본문, OCR, VLM summary 대상
   - `image_dense`: page/figure/map image 대상
   - `caption_dense`: asset caption과 VLM summary 대상
   - Qdrant named vector 또는 collection 분리 전략을 선택 가능하게 둠

7. **Lexical index**
   - BM25 토큰 index 생성
   - 한국어 형태소 분석기는 추후 교체 가능하게 tokenizer interface로 둠
   - 정책명, 지명, 연도, 법령명 같은 exact match 회수를 담당

8. **Graph extraction**
   - VLM/LLM이 `subject, predicate, object` triple 후보 생성
   - 예: `동북권 -> 발전구상 -> 중랑천 상계`, `2030 서울플랜 -> 중심지체계 -> 3도심 7광역중심 12지역중심`
   - triple은 검색 확장, 근거 연결, 지식 그래프 시각화에 사용

9. **Storage**
   - Qdrant: dense vector와 payload 저장
   - BM25: lexical index manifest 저장 또는 별도 검색엔진으로 이전
   - PostgreSQL: documents/pages/chunks/assets/triples 정규화 저장

## Qdrant 설계

초기에는 collection 하나를 사용한다.

- collection: `planning_chunks`
- point id: stable hash
- named vectors:
  - `text_dense`
  - `image_dense`
  - `caption_dense`
- payload:
  - `doc_id`
  - `chunk_id`
  - `page_start`
  - `page_end`
  - `kind`
  - `section`
  - `text_quality`
  - `asset_ids`
  - `source_url`

지도와 페이지 이미지는 이미지 vector가 없는 환경에서도 `caption_dense`로 검색 가능해야 한다.

## PostgreSQL 고려

PostgreSQL은 원본성과 관계형 질의에 사용한다.

- documents: 문서 metadata
- pages: 페이지 profile
- chunks: 검색 단위
- assets: 지도/표/그림 이미지와 캡션
- triples: 그래프 관계

Vector를 PostgreSQL에 바로 넣는 경우를 대비해 `pgvector` extension을 고려하지만, 기본 검색 vector store는 Qdrant로 둔다.

## Retrieval

질의는 다음 결과를 결합한다.

- dense text search
- dense image/caption search
- BM25 lexical search
- graph expansion

초기 결합 방식은 Reciprocal Rank Fusion으로 둔다. 이후 질의 타입을 분류해서 지도 질의는 image/caption 가중치를 높이고, 지명·정책명 질의는 BM25 가중치를 높인다.

## 현재 산출물 패키지

`chunking-docs package`는 DB에 바로 넣기 전 검토 가능한 로컬 패키지를 만든다.

- `manifest.json`: 전체 처리 요약
- `pages.jsonl`: 페이지별 텍스트 품질과 시각 요소 밀도
- `chunks.jsonl`: 검색 단위. 현재는 page-level starter chunk이며 OCR/VLM 결과가 붙으면 세분화한다.
- `assets.jsonl`: 페이지 렌더링 이미지와 지도/표/그림 분류 힌트
- `triples.jsonl`: 목차/section 기반 graph triple 후보
- `bm25_tokens.json`: BM25 토큰 manifest
- `qdrant_text_records.jsonl`: Qdrant upsert dry-run record
- `qdrant_collection.json`: named vector와 payload index 설계

## RTX 5090 활용

GPU는 배치 처리용으로 사용한다.

- 페이지 렌더링 이미지를 batch로 VLM에 투입
- 지도/그래프/표 페이지 우선 처리
- 텍스트 OCR과 VLM summary를 별도 캐시에 저장
- 실패한 페이지는 재시도 가능한 job record로 남김

VLM backend는 모델을 고정하지 않고 interface로 둔다. 로컬에서 사용할 수 있는 Korean-capable VLM 또는 OCR 모델을 바꿔가며 같은 산출 schema에 기록한다.

초기 backend 후보:

- OCR: `TesseractOCRBackend` 또는 추후 PaddleOCR/EasyOCR adapter
- VLM: `HuggingFaceVLMBackend`
- text dense: `SentenceTransformerTextEmbedder`, 기본 후보 `BAAI/bge-m3`
- image dense: `TransformersImageEmbedder`, 기본 후보 `openai/clip-vit-large-patch14`

실제 운영에서는 모델별 결과를 같은 `assets.jsonl`, `chunks.jsonl`, `qdrant_*_records.jsonl` schema에 누적해서 모델 성능을 비교한다.
