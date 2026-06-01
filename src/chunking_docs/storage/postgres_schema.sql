create extension if not exists vector;

create table if not exists documents (
    doc_id text primary key,
    title text not null,
    source_url text,
    local_path text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists pages (
    doc_id text not null references documents(doc_id) on delete cascade,
    page_no integer not null,
    width double precision,
    height double precision,
    text_quality text,
    profile jsonb not null default '{}'::jsonb,
    primary key (doc_id, page_no)
);

create table if not exists chunks (
    chunk_id text primary key,
    doc_id text not null references documents(doc_id) on delete cascade,
    page_start integer not null,
    page_end integer not null,
    kind text not null,
    section jsonb not null default '{}'::jsonb,
    text text not null,
    metadata jsonb not null default '{}'::jsonb
);

create table if not exists chunk_lexical_tokens (
    chunk_id text primary key references chunks(chunk_id) on delete cascade,
    doc_id text not null references documents(doc_id) on delete cascade,
    tokenizer jsonb not null default '{}'::jsonb,
    text_char_count integer not null default 0,
    token_count integer not null default 0,
    tokens text[] not null default '{}'::text[],
    metadata jsonb not null default '{}'::jsonb
);

create table if not exists assets (
    asset_id text primary key,
    doc_id text not null references documents(doc_id) on delete cascade,
    page_no integer not null,
    kind text not null,
    path text,
    bbox double precision[],
    caption text,
    ocr_text text,
    vlm_summary text,
    metadata jsonb not null default '{}'::jsonb
);

create table if not exists chunk_asset_links (
    chunk_id text not null references chunks(chunk_id) on delete cascade,
    asset_id text not null references assets(asset_id) on delete cascade,
    doc_id text not null references documents(doc_id) on delete cascade,
    source text not null,
    metadata jsonb not null default '{}'::jsonb,
    primary key (chunk_id, asset_id)
);

create table if not exists triples (
    triple_id text primary key,
    doc_id text not null references documents(doc_id) on delete cascade,
    chunk_id text not null references chunks(chunk_id) on delete cascade,
    subject text not null,
    predicate text not null,
    object text not null,
    qualifiers jsonb not null default '{}'::jsonb,
    confidence double precision
);

create table if not exists embedding_artifacts (
    doc_id text not null references documents(doc_id) on delete cascade,
    vector_name text not null,
    collection text not null,
    file text not null,
    record_count integer not null default 0,
    dimension integer not null,
    distance text not null,
    note text,
    bytes bigint not null default 0,
    sha256 text,
    metadata jsonb not null default '{}'::jsonb,
    primary key (doc_id, vector_name)
);

create index if not exists chunks_doc_page_idx on chunks(doc_id, page_start, page_end);
create index if not exists pages_text_quality_idx on pages(doc_id, text_quality);
create index if not exists chunks_text_quality_idx on chunks ((metadata->>'text_quality'));
create index if not exists chunk_lexical_tokens_doc_idx on chunk_lexical_tokens(doc_id, chunk_id);
create index if not exists chunk_lexical_tokens_tokens_idx on chunk_lexical_tokens using gin (tokens);
create index if not exists assets_doc_page_idx on assets(doc_id, page_no);
create index if not exists assets_scope_idx on assets ((metadata->>'asset_scope'));
create index if not exists assets_text_quality_idx on assets ((metadata->>'text_quality'));
create index if not exists assets_parent_asset_idx on assets ((metadata->>'parent_asset_id'));
create index if not exists assets_tile_idx on assets(doc_id, page_no, (metadata->>'tile_index'));
create index if not exists chunk_asset_links_asset_idx on chunk_asset_links(asset_id, chunk_id);
create index if not exists chunk_asset_links_doc_idx on chunk_asset_links(doc_id, asset_id);
create index if not exists triples_spo_idx on triples(subject, predicate, object);
create index if not exists chunks_text_bm25_idx on chunks using gin (to_tsvector('simple', text));
create index if not exists embedding_artifacts_collection_idx on embedding_artifacts(collection, vector_name);
