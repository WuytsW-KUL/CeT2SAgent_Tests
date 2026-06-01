#!/usr/bin/env python3
"""
Direct LLM baseline for QALD-9 DBpedia.

Instead of routing through the text2sparql API, this script sends each
question straight to an LLM via OpenRouter (ChatOpenAI-compatible) and
asks it to produce a SPARQL query.  Results are stored in the same
format as evaluate_qald9.py so the two runs are directly comparable.

Usage example:
    python evaluate_llm_direct.py \
        --input-file qald_9_plus_test_dbpedia_104.json \
        --model-name openai/gpt-4o-mini \
        --limit 10
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DEFAULT_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"

SYSTEM_PROMPT = """\
You are an expert SPARQL query generator for DBpedia.

Given a natural-language question, output a single valid SPARQL query 

Rules:
- Return ONLY the raw SPARQL query — no explanation, no markdown fences, \
no extra text whatsoever.
"""

USER_PROMPT_TEMPLATE = "Question: {question}\n\nSPARQL:"


# ──────────────────────────────────────────────
# Logging / IO helpers  (same as evaluate_qald9.py)
# ──────────────────────────────────────────────
def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "evaluation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def flush_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


# ──────────────────────────────────────────────
# Dataset helpers  (same as evaluate_qald9.py)
# ──────────────────────────────────────────────
def load_questions(input_file: Path) -> List[Dict[str, Any]]:
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("questions", [])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dataset format in {input_file}")


def extract_question(question_field: Any, lang: str = "en") -> str:
    if isinstance(question_field, str):
        return question_field.strip()
    if isinstance(question_field, list):
        for item in question_field:
            if item.get("language") == lang and item.get("string"):
                return item["string"].strip()
        for item in question_field:
            if item.get("string"):
                return item["string"].strip()
    if isinstance(question_field, dict):
        return (question_field.get("string") or question_field.get("text") or "").strip()
    return ""


def extract_gold_sparql(item: Dict[str, Any]) -> Optional[str]:
    query_block = item.get("query")
    if isinstance(query_block, dict):
        sparql = query_block.get("sparql")
        return sparql.strip() if sparql else None
    if "sparql" in item and item["sparql"]:
        return str(item["sparql"]).strip()
    return None


# ──────────────────────────────────────────────
# SPARQL execution + result comparison
# (same as evaluate_qald9.py)
# ──────────────────────────────────────────────
def execute_sparql(
    query: Optional[str],
    endpoint: str,
    timeout: int = 60,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not query:
        return None, "No query provided"
    try:
        response = requests.get(
            endpoint,
            params={"query": query, "format": "json"},
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "text2sparql-evaluation/1.0",
            },
            timeout=timeout,
        )
        if response.status_code == 200:
            return response.json(), None
        return None, f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return None, str(e)


def canonicalize_uri(value: str) -> str:
    value = value.strip()
    doubled = "http://dbpedia.org/resource/http://dbpedia.org/resource/"
    if value.startswith(doubled):
        value = value.replace(doubled, "http://dbpedia.org/resource/", 1)
    while value.endswith("_"):
        value = value[:-1]
    return value


def normalize_binding_value(value_obj: Dict[str, Any]) -> str:
    vtype = value_obj.get("type", "")
    value = str(value_obj.get("value", "")).strip()
    if vtype == "uri":
        return canonicalize_uri(value)
    if vtype in {"literal", "typed-literal"}:
        datatype = value_obj.get("datatype")
        lang = value_obj.get("xml:lang") or value_obj.get("lang")
        if datatype:
            return f'"{value}"^^<{datatype}>'
        if lang:
            return f'"{value}"@{lang}'
        return f'"{value}"'
    if vtype == "bnode":
        return f"_:{value}"
    return value


def extract_values(result_json: Optional[Dict[str, Any]]) -> Set[str]:
    if not result_json:
        return set()
    if "boolean" in result_json:
        return {str(bool(result_json["boolean"])).lower()}
    values: Set[str] = set()
    bindings = result_json.get("results", {}).get("bindings", [])
    for row in bindings:
        ordered = [normalize_binding_value(row[v]) for v in sorted(row.keys())]
        if not ordered:
            continue
        values.add(ordered[0] if len(ordered) == 1 else " | ".join(ordered))
    return values


def has_empty_bindings(result_json: Optional[Dict[str, Any]]) -> bool:
    if not result_json:
        return True
    if "boolean" in result_json:
        return False
    return len(result_json.get("results", {}).get("bindings", [])) == 0


def compute_metrics(predicted: Set[str], gold: Set[str]) -> Tuple[int, int, int, float, float, float]:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    if not predicted and not gold:
        return tp, fp, fn, 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return tp, fp, fn, precision, recall, f1


def compute_micro(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f


def compute_macro(ps: List[float], rs: List[float], fs: List[float]) -> Tuple[float, float, float]:
    n = len(fs)
    if n == 0:
        return 0.0, 0.0, 0.0
    return sum(ps) / n, sum(rs) / n, sum(fs) / n


# ──────────────────────────────────────────────
# LLM call
# ──────────────────────────────────────────────
STANDARD_PREFIXES: Dict[str, str] = {
    "rdf":     "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs":    "http://www.w3.org/2000/01/rdf-schema#",
    "owl":     "http://www.w3.org/2002/07/owl#",
    "xsd":     "http://www.w3.org/2001/XMLSchema#",
    "foaf":    "http://xmlns.com/foaf/0.1/",
    "dbo":     "http://dbpedia.org/ontology/",
    "dbp":     "http://dbpedia.org/property/",
    "dbr":     "http://dbpedia.org/resource/",
    "res":     "http://dbpedia.org/resource/",
    "dbc":     "http://dbpedia.org/resource/Category:",
    "dbd":     "http://dbpedia.org/datatype/",
    "dbpedia": "http://dbpedia.org/",
    "schema":  "http://schema.org/",
    "geo":     "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "skos":    "http://www.w3.org/2004/02/skos/core#",
    "dc":      "http://purl.org/dc/elements/1.1/",
    "dct":     "http://purl.org/dc/terms/",
    "yago":    "http://dbpedia.org/class/yago/",
}


def inject_missing_prefixes(query: str) -> str:
    """Prepend any PREFIX declarations that are used in the query but not yet declared."""
    # collect already-declared prefixes  (PREFIX foo: <...>)
    declared = set(re.findall(r"PREFIX\s+(\w+)\s*:", query, flags=re.IGNORECASE))
    # collect used short-form prefixes  (foo:Something  or  foo: <blank-node trick>)
    used = set(re.findall(r"\b(\w+):", query))
    missing = used - declared
    lines = [
        f"PREFIX {prefix}: <{uri}>"
        for prefix, uri in STANDARD_PREFIXES.items()
        if prefix in missing
    ]
    if not lines:
        return query
    return "\n".join(lines) + "\n" + query


def extract_sparql_from_response(text: str) -> str:
    """Strip markdown code fences, inject missing prefixes, return clean query."""
    # Remove ```sparql ... ``` or ``` ... ```
    text = re.sub(r"```(?:sparql)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    # Strip the "SPARQL:" prefix if the model echoed it
    text = re.sub(r"^SPARQL:\s*", "", text.strip(), flags=re.IGNORECASE)
    text = text.strip()
    return inject_missing_prefixes(text)


def call_llm(
    llm: ChatOpenAI,
    question_text: str,
) -> Tuple[Optional[str], Optional[str], int, int]:
    """
    Returns (sparql_query, error, prompt_tokens, completion_tokens).
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT_TEMPLATE.format(question=question_text)),
    ]
    try:
        response = llm.invoke(messages)
        raw = response.content or ""
        usage = getattr(response, "usage_metadata", None) or {}
        prompt_tokens     = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        sparql = extract_sparql_from_response(raw)
        if not sparql:
            return None, "LLM returned empty response", prompt_tokens, completion_tokens
        return sparql, None, prompt_tokens, completion_tokens
    except Exception as e:
        return None, str(e), 0, 0


# ──────────────────────────────────────────────
# Summary builder
# ──────────────────────────────────────────────
def make_summary(
    total: int,
    processed: int,
    skipped: int,
    tp: int, fp: int, fn: int,
    pps: List[float], prs: List[float], pfs: List[float],
    errors_count: int,
    started_at: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost: float = 0.0,
) -> Dict[str, Any]:
    micro_p, micro_r, micro_f = compute_micro(tp, fp, fn)
    macro_p, macro_r, macro_f = compute_macro(pps, prs, pfs)
    return {
        "total_questions":         total,
        "processed_questions":     processed,
        "skipped_questions":       skipped,
        "remaining_questions":     total - processed - skipped,
        "micro_precision":         micro_p,
        "micro_recall":            micro_r,
        "micro_f1":                micro_f,
        "macro_precision":         macro_p,
        "macro_recall":            macro_r,
        "macro_f1":                macro_f,
        "questions_with_f1_lt_1":  errors_count,
        "elapsed_seconds":         time.time() - started_at,
        "done":                    (processed + skipped) == total,
        "prompt_tokens":           prompt_tokens,
        "completion_tokens":       completion_tokens,
        "total_cost_usd":          cost,
    }


# ──────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────
def evaluate(
    input_file: Path,
    output_dir: Path,
    sparql_endpoint: str,
    lang: str,
    limit: Optional[int],
    sleep_seconds: float,
    model_name: str,
    temperature: float,
    cost_prompt: float,
    cost_completion: float,
    start_at_index: int,
    end_at_index: Optional[int],
) -> None:
    llm = ChatOpenAI(
        model=model_name,
        api_key="",
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
    )

    questions_full = load_questions(input_file)
    total_questions = len(questions_full)

    output_file  = output_dir / "results.json"
    error_file   = output_dir / "errors.json"
    skipped_file = output_dir / "skipped_empty_gold.json"
    summary_file = output_dir / "summary.json"

    results:  List[Dict[str, Any]] = []
    errors:   List[Dict[str, Any]] = []
    skipped:  List[Dict[str, Any]] = []

    running_tp = running_fp = running_fn = 0
    processed = skipped_count = 0
    total_prompt_tokens = total_completion_tokens = 0
    total_cost = 0.0
    pps: List[float] = []
    prs: List[float] = []
    pfs: List[float] = []

    started_at = time.time()

    # Initialise empty output files so they always exist
    for path, obj in [(output_file, results), (error_file, errors),
                      (skipped_file, skipped)]:
        flush_json(path, obj)
    flush_json(summary_file, make_summary(total_questions, 0, 0, 0, 0, 0, [], [], [],
                                          0, started_at))

    questions = questions_full[start_at_index:end_at_index]
    if limit is not None:
        questions = questions[:limit]

    for i, item in enumerate(questions, start=start_at_index + 1):
        question_text = extract_question(item.get("question"), lang=lang)
        gold_query    = extract_gold_sparql(item)
        qid           = item.get("id", i)

        logging.info("(%d/%d) QID=%s | %s", i, total_questions, qid, question_text)

        # ── 1. Execute gold query ──────────────────────────────────────
        gold_result, gold_error = (None, "No gold query provided")
        if gold_query:
            gold_result, gold_error = execute_sparql(gold_query, sparql_endpoint)
        gold_values = extract_values(gold_result)

        # ── 2. Skip if gold result is empty ───────────────────────────
        if has_empty_bindings(gold_result):
            skipped_count += 1
            entry = {
                "index": i, "id": qid, "question": question_text,
                "gold_query": gold_query, "gold_result": gold_result,
                "gold_execution_error": gold_error,
                "reason": "gold query returned empty result",
                "gold_values": sorted(gold_values),
            }
            skipped.append(entry)
            flush_json(skipped_file, skipped)
            flush_json(summary_file, make_summary(
                total_questions, processed, skipped_count,
                running_tp, running_fp, running_fn, pps, prs, pfs,
                len(errors), started_at, total_prompt_tokens,
                total_completion_tokens, total_cost,
            ))
            print(f"\n[{i}/{total_questions}] {question_text}")
            print("  Skipped — gold query returned empty result")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            continue

        # ── 3. Call LLM directly ───────────────────────────────────────
        t0 = time.time()
        generated_query, llm_error, prompt_tok, completion_tok = call_llm(llm, question_text)
        llm_time = time.time() - t0

        # ── 4. Execute generated query ─────────────────────────────────
        generated_result, exec_error = (None, None)
        if generated_query:
            generated_result, exec_error = execute_sparql(generated_query, sparql_endpoint)
        generated_error = llm_error or exec_error
        generated_values = extract_values(generated_result)

        # ── 5. Metrics ─────────────────────────────────────────────────
        tp, fp, fn, precision, recall, f1 = compute_metrics(generated_values, gold_values)
        running_tp += tp; running_fp += fp; running_fn += fn
        processed += 1

        q_cost = (
            prompt_tok * cost_prompt / 1_000_000
            + completion_tok * cost_completion / 1_000_000
        )
        total_prompt_tokens     += prompt_tok
        total_completion_tokens += completion_tok
        total_cost              += q_cost

        pps.append(precision); prs.append(recall); pfs.append(f1)

        micro_p, micro_r, micro_f = compute_micro(running_tp, running_fp, running_fn)
        macro_p, macro_r, macro_f = compute_macro(pps, prs, pfs)

        # ── 6. Persist ─────────────────────────────────────────────────
        result_entry = {
            "index":                    i,
            "id":                       qid,
            "question":                 question_text,
            "generated_query":          generated_query,
            "generated_execution_error": generated_error,
            "gold_query":               gold_query,
            "gold_execution_error":     gold_error,
            "llm_time_seconds":         round(llm_time, 3),
            "f1":                       f1,
            "precision":                precision,
            "recall":                   recall,
            "running_micro_f1":         micro_f,
            "running_macro_f1":         macro_f,
            "prompt_tokens":            prompt_tok,
            "completion_tokens":        completion_tok,
            "cost_usd":                 q_cost,
            "generated_values":         sorted(generated_values),
            "gold_values":              sorted(gold_values),
            "only_in_generated":        sorted(generated_values - gold_values),
            "only_in_gold":             sorted(gold_values - generated_values),
        }
        results.append(result_entry)
        flush_json(output_file, results)
        if f1 < 1.0:
            errors.append(result_entry)
            flush_json(error_file, errors)

        flush_json(summary_file, make_summary(
            total_questions, processed, skipped_count,
            running_tp, running_fp, running_fn, pps, prs, pfs,
            len(errors), started_at, total_prompt_tokens,
            total_completion_tokens, total_cost,
        ))

        # ── 7. Console output ──────────────────────────────────────────
        logging.info(
            "QID=%s done | P=%.4f R=%.4f F1=%.4f | gold=%d pred=%d | %.1fs",
            qid, precision, recall, f1,
            len(gold_values), len(generated_values), llm_time,
        )
        print(f"\n[{i}/{total_questions}] {question_text}")
        print(f"  Generated : {generated_query or '(none)'}")
        print(f"  Q P/R/F1  : {precision:.4f} / {recall:.4f} / \033[33m{f1:.4f}\033[0m")
        print(f"  Running µF1: {micro_f:.4f}  mF1: \033[33m{macro_f:.4f}\033[0m")
        if generated_error:
            print(f"  Error     : {generated_error}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    # ── Final summary ──────────────────────────────────────────────────
    final = make_summary(
        total_questions, processed, skipped_count,
        running_tp, running_fp, running_fn, pps, prs, pfs,
        len(errors), started_at, total_prompt_tokens,
        total_completion_tokens, total_cost,
    )
    flush_json(summary_file, final)

    print("\n" + "=" * 60)
    print("DONE")
    print(json.dumps(final, indent=4, ensure_ascii=False))
    print(f"\nResults  → {output_file}")
    print(f"Errors   → {error_file}")
    print(f"Summary  → {summary_file}")
    print(
        f"\nTokens   → input: {total_prompt_tokens:,}  |  "
        f"output: {total_completion_tokens:,}  |  "
        f"total: {total_prompt_tokens + total_completion_tokens:,}"
    )


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct-LLM SPARQL baseline for QALD-9 DBpedia"
    )
    parser.add_argument("--input-file",       type=Path, default=Path("qald_9_plus_test_dbpedia.json"),
                        help="Path to QALD JSON file")
    parser.add_argument("--sparql-endpoint",  type=str,  default=DEFAULT_SPARQL_ENDPOINT)
    parser.add_argument("--lang",             type=str,  default="en")
    parser.add_argument("--limit",            type=int,  default=None,
                        help="Max number of questions to process")
    parser.add_argument("--sleep-seconds",    type=float, default=0.0)
    parser.add_argument("--model-name",       type=str,  default="openai/gpt-4o-mini",
                        help="OpenRouter model identifier")
    parser.add_argument("--temp",             type=float, default=0.0)
    parser.add_argument("--cost-prompt",      type=float, default=2.50,
                        help="Cost per 1M prompt tokens (USD)")
    parser.add_argument("--cost-completion",  type=float, default=15.0,
                        help="Cost per 1M completion tokens (USD)")
    parser.add_argument("--start-at-index",   type=int,  default=0)
    parser.add_argument("--end-at-index",     type=int,  default=None)
    parser.add_argument("--test-name",        type=str,  default=None,
                        help="Optional suffix appended to the output folder name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()


    timestamp   = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_slug  = args.model_name.replace("/", "-").replace(":", "-")
    folder_name = (
        f"evaluation_{Path(args.input_file).stem}_{timestamp}"
        f"_{args.lang}_{model_slug}_direct_llm"
    )
    if args.test_name:
        folder_name += f"_{args.test_name}"

    output_dir = Path("test_results") / folder_name
    setup_logging(output_dir)

    logging.info("Model        : %s", args.model_name)
    logging.info("Input file   : %s", args.input_file)
    logging.info("Output dir   : %s", output_dir)

    evaluate(
        input_file      = args.input_file,
        output_dir      = output_dir,
        sparql_endpoint = args.sparql_endpoint,
        lang            = args.lang,
        limit           = args.limit,
        sleep_seconds   = args.sleep_seconds,
        model_name      = args.model_name,
        temperature     = args.temp,
        cost_prompt     = args.cost_prompt,
        cost_completion = args.cost_completion,
        start_at_index  = args.start_at_index,
        end_at_index    = args.end_at_index,
    )


if __name__ == "__main__":
    main()
