"""High-throughput asynchronous synthetic clinical corpus generator (vLLM + RAG + SBERT filters).

Architecture:
  - Pure asyncio with persistent httpx connection pooling across endpoints.
  - Intra-source RAG retrieval and triple SBERT semantic filtering
    (copy threshold, divergence threshold, and pairwise redundancy).
  - Exponential-backoff retries and periodic document-level checkpoints.
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.asyncio import tqdm as async_tqdm

# ─────────────────────────────────────────────────────────────────
# PARÁMETROS CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser(description="Synthetic clinical corpus generation pipeline (vLLM + RAG + SBERT)")
    ap.add_argument("--normalized-corpus", required=True, help="Path to normalized seed JSON")
    ap.add_argument("--output-prefix", required=True, help="Output file prefix (e.g. corpus/sinteticos_qwen3.8_27b/sintetico_cantemist)")
    ap.add_argument("--endpoints", nargs="+", default=["http://localhost:8000/v1"], help="vLLM OpenAI-compatible base URLs")
    ap.add_argument("--model-path", default="Qwen/Qwen3-8B", help="Served model name or path (e.g. qwen3_8b or qwen3.8_27b)")
    ap.add_argument("--api-key", default="placeholder", help="API key for external endpoints")
    ap.add_argument("--n-per-doc", type=int, default=10, help="Target synthetic variants per seed document")
    ap.add_argument("--max-concurrent", type=int, default=16, help="Maximum concurrent seed documents")
    ap.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling")
    ap.add_argument("--max-intentos", type=int, default=5, help="Maximum generation rounds per seed document")
    ap.add_argument("--max-lote", type=int, default=15, help="Maximum candidate requests per round")
    ap.add_argument("--k-retrieved", type=int, default=3, help="Number of RAG exemplars retrieved per seed")
    ap.add_argument("--umbral-copia", type=float, default=0.90, help="Cosine similarity > threshold rejected as copy")
    ap.add_argument("--umbral-divergencia", type=float, default=0.40, help="Cosine similarity < threshold rejected as off-domain")
    ap.add_argument("--umbral-pairwise", type=float, default=0.95, help="Cosine similarity > threshold rejected as redundant")
    ap.add_argument("--sbert-model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", help="Sentence-Transformer model for RAG/filtering")
    ap.add_argument("--max-model-len", type=int, default=16384, help="Maximum model context length in tokens")
    ap.add_argument("--checkpoint-every", type=int, default=5, help="Save checkpoint every N documents")
    ap.add_argument("--max-retries", type=int, default=3, help="Retry attempts for transient HTTP errors")
    ap.add_argument("--backoff-base", type=float, default=2.0, help="Exponential backoff base in seconds")
    ap.add_argument("--no-enable-thinking", action="store_true", help="Disable Qwen chat_template_kwargs field")
    ap.add_argument("--no-rag-filter-by-source", action="store_true", help="Retrieve RAG examples across all sources")
    ap.add_argument("--seed", type=int, default=14, help="Sampling RNG seed (default: 14)")
    ap.add_argument("--source-filter", default=None, help="Filter normalized documents by source name")
    ap.add_argument("--max-docs", type=int, default=None, help="Optional cap on seed documents processed")
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────
# PROMPTS Y FORMATEO
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un asistente que genera textos clínicos sintéticos en español latinoamericano.
Los textos son registros médicos escritos por médicos o administrativos.
Características del estilo real:
- Pueden ser diagnósticos, notas clínicas, o descripciones de casos
- Estilo informal y clínico
- Abreviaciones médicas comunes
- Sin puntuación final innecesaria
- En español clínico latinoamericano
Genera SOLO el texto clínico, sin comillas, sin explicación."""


def build_prompt(example_text: str, retrieved: List[str], max_model_len: int) -> str:
    words = example_text.split()
    if len(words) > 500:
        example_text = " ".join(words[:500])
    largo_base = len(example_text.split())
    min_palabras = max(int(largo_base * 0.7), 40)
    max_palabras = min(int(largo_base * 1.3), 600)

    retrieved_parts = []
    if retrieved:
        for ret in retrieved[:2]:
            r_words = ret.split()
            if len(r_words) > 250:
                r_words = r_words[:250]
            retrieved_parts.append(" ".join(r_words))

    if retrieved_parts:
        block = "\n".join(f"  {i+1}. {ex}" for i, ex in enumerate(retrieved_parts))
        context = f"Ejemplo principal ({largo_base} palabras):\n{example_text}\n\nEjemplos adicionales del mismo tipo clínico:\n{block}"
    else:
        context = f"Ejemplo ({largo_base} palabras):\n{example_text}"

    return f"""Ejemplos reales de registros médicos en español:

{context}

Genera UN nuevo texto clínico sintético que:
1. Sea del mismo tipo clínico que los ejemplos
2. Siga el mismo estilo informal y clínico
3. NO sea una copia de ninguno de los ejemplos anteriores
4. Tenga entre {min_palabras} y {max_palabras} palabras — esto es OBLIGATORIO

LONGITUD OBLIGATORIA: El texto debe tener entre {min_palabras} y {max_palabras} palabras.
Desarrolla la historia clínica con el mismo nivel de detalle que el ejemplo.
Textos más cortos o excesivamente breves no son válidos.

Escribe solo el texto clínico:"""


def clean_response(raw: str) -> str:
    if "<think>" in raw and "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    if "```" in raw:
        raw = re.sub(r'```[\w]*\n?', '', raw)
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
        raw = raw[1:-1].strip()
    return raw


# ─────────────────────────────────────────────────────────────────
# ÍNDICE RAG Y SBERT (THREAD-SAFE & ASYNC-FRIENDLY)
# ─────────────────────────────────────────────────────────────────

class AsyncCorpusIndex:
    def __init__(self, sbert_model_name: str):
        self.model_name = sbert_model_name
        self.texts: List[str] = []
        self.sources: List[Optional[str]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._sbert = None
        self._lock = threading.Lock()

    def build(self, texts: List[str], sources: Optional[List[str]] = None):
        from sentence_transformers import SentenceTransformer
        self.texts = texts
        self.sources = sources if sources is not None else [None] * len(texts)
        print(f"Indexing corpus ({len(texts)} texts) with SBERT ({self.model_name})...")
        self._sbert = SentenceTransformer(self.model_name)
        self.embeddings = self._sbert.encode(
            texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64
        )
        print(f"  Indexing complete: {self.embeddings.shape}\n")

    def embed_sync(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        with self._lock:
            return self._sbert.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    async def embed_async(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        return await asyncio.to_thread(self.embed_sync, texts)

    def retrieve(self, query_emb: np.ndarray, k: int = 3, filter_source: Optional[str] = None, exclude_index: Optional[int] = None) -> List[str]:
        if filter_source and any(s is not None for s in self.sources):
            valid_indices = [i for i, s in enumerate(self.sources) if str(s) == str(filter_source) and i != exclude_index]
        else:
            valid_indices = [i for i in range(len(self.texts)) if i != exclude_index]

        if not valid_indices:
            return []

        search_embeddings = self.embeddings[valid_indices]
        search_texts = [self.texts[i] for i in valid_indices]
        sims = cosine_similarity(query_emb.reshape(1, -1), search_embeddings)[0]
        sorted_idx = np.argsort(sims)[::-1]
        return [search_texts[i] for i in sorted_idx[:k]]


# ─────────────────────────────────────────────────────────────────
# POOL DE CLIENTES HTTPX ASÍNCRONOS DIRECTOS
# ─────────────────────────────────────────────────────────────────

class EndpointPool:
    def __init__(self, endpoints: List[str], api_key: str):
        self.endpoints = endpoints
        self.api_key = api_key
        self._clients: Dict[str, httpx.AsyncClient] = {}
        self._index = 0

    def init_clients(self):
        for url in self.endpoints:
            self._clients[url] = httpx.AsyncClient(
                base_url=url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=httpx.Timeout(180.0, connect=20.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=50, keepalive_expiry=60.0),
            )

    async def close(self):
        for client in self._clients.values():
            await client.aclose()

    def get_endpoint_client(self) -> Tuple[str, httpx.AsyncClient]:
        url = self.endpoints[self._index % len(self.endpoints)]
        self._index += 1
        return url, self._clients[url]


# ─────────────────────────────────────────────────────────────────
# GENERADOR Y FILTRO DE CANDIDATOS
# ─────────────────────────────────────────────────────────────────

async def generate_single(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    no_enable_thinking: bool,
    max_retries: int,
    backoff_base: float,
    seed: Optional[int] = None,
) -> Optional[str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": 1536,
    }
    if seed is not None:
        payload["seed"] = seed
    if not no_enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    for attempt in range(max_retries):
        try:
            resp = await client.post("/chat/completions", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return clean_response(content)
            elif resp.status_code == 404:
                err_msg = f"Fatal vLLM 404 error: model '{model}' not found on endpoint. {resp.text[:160]}"
                print(err_msg, file=sys.stderr, flush=True)
                raise RuntimeError(err_msg)
            elif resp.status_code == 429 or resp.status_code >= 500:
                print(f"[WARN] vLLM transient error {resp.status_code}: {resp.text[:120]}", file=sys.stderr, flush=True)
                await asyncio.sleep(backoff_base ** attempt)
            else:
                print(f"[ERROR] vLLM client error {resp.status_code}: {resp.text[:160]}", file=sys.stderr, flush=True)
                return None
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[ERROR] Endpoint connection exception: {e}", file=sys.stderr, flush=True)
            await asyncio.sleep(backoff_base ** attempt)
    return None


async def generate_batch_async(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    n: int,
    temperature: float,
    top_p: float,
    no_enable_thinking: bool,
    max_retries: int,
    backoff_base: float,
    seed: Optional[int] = None,
) -> List[str]:
    tasks = [
        generate_single(
            client, model, prompt, temperature, top_p, no_enable_thinking, max_retries, backoff_base,
            seed=(seed + i if seed is not None else None)
        )
        for i in range(n)
    ]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]


async def filter_candidates_async(
    candidates: List[str],
    base_embedding: np.ndarray,
    index: AsyncCorpusIndex,
    umbral_copia: float,
    umbral_divergencia: float,
    umbral_pairwise: float,
    existing_valid: Optional[List[str]] = None,
    existing_embeddings: Optional[np.ndarray] = None,
) -> Tuple[List[str], np.ndarray, dict]:
    empty = {"total": 0, "copias": 0, "divergentes": 0, "quasi_copias": 0}
    if not candidates:
        return [], np.array([]), empty

    stats = {"total": len(candidates), "copias": 0, "divergentes": 0, "quasi_copias": 0}
    candidate_embeddings = await index.embed_async(candidates)

    # 1. Similitud vs base
    sims = cosine_similarity(base_embedding.reshape(1, -1), candidate_embeddings)[0]
    mask = np.ones(len(candidates), dtype=bool)
    for i, sim in enumerate(sims):
        if sim > umbral_copia:
            mask[i] = False
            stats["copias"] += 1
        elif sim < umbral_divergencia:
            mask[i] = False
            stats["divergentes"] += 1

    tras_f1 = [c for c, v in zip(candidates, mask) if v]
    emb_tras_f1 = candidate_embeddings[mask]
    if not tras_f1:
        return [], np.array([]), stats

    # 2. Pairwise lote actual
    if len(tras_f1) > 1:
        pw = cosine_similarity(emb_tras_f1)
        kept = [0]
        for i in range(1, len(tras_f1)):
            if max(pw[i][j] for j in kept) > umbral_pairwise:
                stats["quasi_copias"] += 1
            else:
                kept.append(i)
        tras_f2 = [tras_f1[i] for i in kept]
        emb_tras_f2 = emb_tras_f1[kept]
    else:
        tras_f2 = tras_f1
        emb_tras_f2 = emb_tras_f1

    # 3. Pairwise vs existentes
    if existing_valid and tras_f2:
        prev_emb = existing_embeddings if (existing_embeddings is not None and len(existing_embeddings) > 0) else await index.embed_async(existing_valid)
        final, final_emb = [], []
        for cand, emb in zip(tras_f2, emb_tras_f2):
            if np.max(cosine_similarity(emb.reshape(1, -1), prev_emb)[0]) > umbral_pairwise:
                stats["quasi_copias"] += 1
            else:
                final.append(cand)
                final_emb.append(emb)
        return final, np.array(final_emb) if final_emb else np.array([]), stats

    return tras_f2, emb_tras_f2, stats


# ─────────────────────────────────────────────────────────────────
# WORKER POR DOCUMENTO
# ─────────────────────────────────────────────────────────────────

async def process_document(
    doc_tuple: Tuple[int, str, Optional[str]],
    sem: asyncio.Semaphore,
    index: AsyncCorpusIndex,
    pool: EndpointPool,
    args,
) -> dict:
    source_index, source_text, source_source = doc_tuple

    async with sem:
        endpoint_url, client = pool.get_endpoint_client()

        # Obtener embedding del texto base y retrieval RAG
        base_embedding = (await index.embed_async([source_text]))[0]
        filter_source = source_source if not args.no_rag_filter_by_source else None
        retrieved = index.retrieve(
            base_embedding,
            k=args.k_retrieved,
            filter_source=filter_source,
            exclude_index=source_index,
        )
        prompt = build_prompt(source_text, retrieved, args.max_model_len)

        valid_texts = []
        valid_embeddings = np.array([])
        total_stats = {"total": 0, "copias": 0, "divergentes": 0, "quasi_copias": 0}
        intentos = 0

        while len(valid_texts) < args.n_per_doc and intentos < args.max_intentos:
            intentos += 1
            faltan = args.n_per_doc - len(valid_texts)
            lote_actual = min(max(faltan * 2, 5), args.max_lote)

            batch_seed = (args.seed + source_index * 1000 + intentos * 100) if getattr(args, 'seed', None) is not None else None

            candidates = await generate_batch_async(
                client=client,
                model=args.model_path,
                prompt=prompt,
                n=lote_actual,
                temperature=args.temperature,
                top_p=args.top_p,
                no_enable_thinking=args.no_enable_thinking,
                max_retries=args.max_retries,
                backoff_base=args.backoff_base,
                seed=batch_seed,
            )

            new_valid, new_emb, stats = await filter_candidates_async(
                candidates=candidates,
                base_embedding=base_embedding,
                index=index,
                umbral_copia=args.umbral_copia,
                umbral_divergencia=args.umbral_divergencia,
                umbral_pairwise=args.umbral_pairwise,
                existing_valid=valid_texts,
                existing_embeddings=valid_embeddings,
            )

            valid_texts.extend(new_valid)
            if len(new_emb) > 0:
                valid_embeddings = new_emb if len(valid_embeddings) == 0 else np.vstack([valid_embeddings, new_emb])

            for k in total_stats:
                total_stats[k] += stats.get(k, 0)

        completado = len(valid_texts) >= args.n_per_doc
        return {
            "source_index": source_index,
            "source_text": source_text,
            "source_source": source_source,
            "generated_texts": valid_texts[:args.n_per_doc],
            "n_valid": len(valid_texts[:args.n_per_doc]),
            "completado": completado,
            "n_intentos": intentos,
            "retrieved": retrieved,
            "filter_stats": total_stats,
        }


# ─────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────

async def run_pipeline_async(args):
    json_path = f"{args.output_prefix}.json"
    csv_path = f"{args.output_prefix}.csv"
    ckpt_path = f"{args.output_prefix}_checkpoint.json"
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SYNTHETIC CLINICAL GENERATION PIPELINE (ASYNC vLLM)")
    print(f"Corpus:          {args.normalized_corpus}")
    print(f"Model:           {args.model_path}")
    print(f"Endpoints:       {args.endpoints}")
    print(f"Concurrency:     {args.max_concurrent} seed documents")
    print(f"Target N/doc:    {args.n_per_doc}")
    print("=" * 60)

    from corpus_normalizer import load_normalized
    print("Loading normalized seed corpus...")
    normalized = load_normalized(args.normalized_corpus)
    all_texts = [row["text"] for row in normalized]
    all_sources = [row.get("source_detail") or row.get("source") for row in normalized]

    if getattr(args, "source_filter", None):
        indices = [i for i, s in enumerate(all_sources) if str(s) == args.source_filter]
        all_texts = [all_texts[i] for i in indices]
        all_sources = [all_sources[i] for i in indices]
        print(f"  Source '{args.source_filter}': {len(all_texts)} documents")
    else:
        from collections import Counter
        print(f"  Total seed documents: {len(all_texts)}")
        for src, count in Counter(all_sources).most_common():
            print(f"    {src}: {count}")

    # Build RAG index over full source before optional --max-docs slicing
    index = AsyncCorpusIndex(args.sbert_model)
    index.build(all_texts, all_sources)

    if args.max_docs:
        all_texts = all_texts[:args.max_docs]
        all_sources = all_sources[:args.max_docs]

    # Resume from checkpoint if present
    results = []
    processed_indices = set()
    ckpt_file = Path(ckpt_path)
    if ckpt_file.exists():
        try:
            results = json.loads(ckpt_file.read_text(encoding="utf-8"))
            processed_indices = {r["source_index"] for r in results}
            print(f"[OK] Resuming from checkpoint: {len(results)} documents already processed.")
        except Exception:
            pass

    pending = [
        (i, text, source)
        for i, (text, source) in enumerate(zip(all_texts, all_sources))
        if i not in processed_indices
    ]
    # Process longer seed notes first for balanced throughput
    pending = sorted(pending, key=lambda x: len(x[1].split()), reverse=True)

    pool = EndpointPool(args.endpoints, args.api_key)
    pool.init_clients()

    # Verify connectivity and resolve served model identifier on vLLM endpoint
    try:
        _, test_client = pool.get_endpoint_client()
        resp = await test_client.get("/models")
        if resp.status_code == 200:
            m_list = [m["id"] for m in resp.json().get("data", [])]
            if args.model_path not in m_list:
                cand = None
                t_clean = Path(args.model_path).name.lower().replace("-", "_")
                for m_id in m_list:
                    m_clean = Path(m_id).name.lower().replace("-", "_")
                    if m_clean == t_clean or t_clean in m_clean or m_clean in t_clean:
                        cand = m_id
                        break
                if cand:
                    print(f"[OK] Resolved served model name from endpoint: '{cand}' (requested: '{args.model_path}')")
                    args.model_path = cand
                elif len(m_list) == 1:
                    print(f"[OK] Using single served model on vLLM: '{m_list[0]}' (requested: '{args.model_path}')")
                    args.model_path = m_list[0]
                else:
                    print(f"[WARN] Model '{args.model_path}' not in /models. Available: {m_list}")
            else:
                print(f"[OK] Verified served model on endpoint: '{args.model_path}'")
    except Exception as e:
        print(f"[WARN] Could not query /models endpoint: {e}")

    sem = asyncio.Semaphore(args.max_concurrent)

    total_valid = sum(r["n_valid"] for r in results)
    incomplete = sum(1 for r in results if not r.get("completado", True))

    print(f"\nProcessing {len(pending)} pending seed documents (concurrency: {args.max_concurrent})...\n")

    tasks = [process_document(doc_tup, sem, index, pool, args) for doc_tup in pending]

    pbar = async_tqdm(total=len(pending), desc="Documents")
    for coro in asyncio.as_completed(tasks):
        res = await coro
        results.append(res)
        total_valid += res["n_valid"]
        if not res["completado"]:
            incomplete += 1
        pbar.update(1)

        if len(results) % args.checkpoint_every == 0:
            tmp_ckpt = f"{ckpt_path}.tmp"
            with open(tmp_ckpt, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
            os.replace(tmp_ckpt, ckpt_path)

    pbar.close()
    await pool.close()

    print(f"\n{'=' * 60}")
    print("GENERATION SUMMARY")
    print(f"  Seed documents:   {len(results)}")
    print(f"  Valid synthetics: {total_valid}")
    print(f"  Incomplete seeds: {incomplete} ({incomplete/max(len(results), 1)*100:.1f}%)")
    print(f"{'=' * 60}\n")

    all_synthetic = [
        {
            "text": text,
            "source_text": r["source_text"],
            "source_source": r["source_source"],
            "source_index": r["source_index"],
        }
        for r in results
        for text in r["generated_texts"]
    ]

    if not all_synthetic:
        raise RuntimeError(
            f"Fatal error: 0 valid synthetic texts generated for {args.output_prefix}. "
            "Aborting without writing empty output files."
        )

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model_path,
            "endpoints": args.endpoints,
            "corpus_input": args.normalized_corpus,
            "n_source_docs": len(all_texts),
            "n_per_doc_target": args.n_per_doc,
            "k_retrieved": args.k_retrieved,
            "max_intentos": args.max_intentos,
            "umbral_copia": args.umbral_copia,
            "umbral_divergencia": args.umbral_divergencia,
            "umbral_pairwise": args.umbral_pairwise,
            "n_synthetic_total": len(all_synthetic),
            "n_incompletos": incomplete,
        },
        "documents": results,
        "synthetic": all_synthetic,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "source_source"])
        writer.writeheader()
        for row in all_synthetic:
            writer.writerow({"text": row["text"], "source_source": row["source_source"]})

    ckpt_file.unlink(missing_ok=True)
    print(f"[OK] JSON: {json_path}")
    print(f"[OK] CSV:  {csv_path}")


def main():
    args = _parse_args()
    asyncio.run(run_pipeline_async(args))


if __name__ == "__main__":
    main()
