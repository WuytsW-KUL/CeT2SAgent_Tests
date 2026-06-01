#!/usr/bin/env python3
"""
QALD evaluator — all languages, question-interleaved order.

For each question, evaluates every available language before moving
to the next question. Output layout:

  test_results/<run>/
    summary_all_languages.json   <- combined metrics across all languages
    evaluation.log
    en/
      results.json
      errors.json
      skipped_empty_gold.json
      summary.json
    de/
      ...
    ...
"""
import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse

import requests


DEFAULT_API_URL = "http://localhost:8000/api"
DEFAULT_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"


def build_api_urls(base_url: str, num_ports: int, low_port: Optional[int] = None) -> List[str]:
    """Return a list of API URLs for ports base_port .. base_port+num_ports-1.

    If *low_port* is given it overrides whatever port is in *base_url*.
    """
    parsed = urlparse(base_url)
    base_port = low_port if low_port is not None else (parsed.port or 8000)
    hostname = parsed.hostname or "localhost"
    return [
        urlunparse(parsed._replace(netloc=f"{hostname}:{base_port + i}"))
        for i in range(num_ports)
    ]


def _call_api_and_execute(
    api_url: str,
    question_text: str,
    sparql_endpoint: str,
    model_name: str,
    log_calls: bool,
    use_translate: bool,
    use_llm_translate: bool = False,
) -> Dict[str, Any]:
    """Worker: call the generation API then execute the returned SPARQL query."""
    generated_query, api_error, translated_question, \
        api_prompt_tokens, api_completion_tokens, api_requests, step_times = call_generation_api(
            api_url=api_url,
            question_text=question_text,
            dataset_url=sparql_endpoint,
            model_name=model_name,
            log_calls=log_calls,
            use_translate=use_translate,
            use_llm_translate=use_llm_translate,
        )

    generated_result = None
    if generated_query:
        generated_result, exec_error = execute_sparql(generated_query, sparql_endpoint)
        if exec_error:
            api_error = exec_error

    return {
        "generated_query": generated_query,
        "generated_error": api_error,
        "translated_question": translated_question,
        "api_prompt_tokens": api_prompt_tokens,
        "api_completion_tokens": api_completion_tokens,
        "api_requests": api_requests,
        "step_times": step_times,
        "generated_result": generated_result,
        "generated_values": extract_values(generated_result),
    }


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


def load_questions(input_file: Path) -> List[Dict[str, Any]]:
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get("questions", [])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dataset format in {input_file}")


def detect_languages(questions: List[Dict[str, Any]]) -> List[str]:
    langs: Set[str] = set()
    for item in questions:
        question_field = item.get("question")
        if isinstance(question_field, list):
            for entry in question_field:
                lang = entry.get("language")
                if lang and entry.get("string", "").strip():
                    langs.add(lang)
    return sorted(langs)


def extract_question(question_field: Any, lang: str) -> str:
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


def has_language(question_field: Any, lang: str) -> bool:
    if isinstance(question_field, list):
        for item in question_field:
            if item.get("language") == lang and item.get("string", "").strip():
                return True
    return False


def extract_gold_sparql(item: Dict[str, Any]) -> Optional[str]:
    query_block = item.get("query")

    if isinstance(query_block, dict):
        sparql = query_block.get("sparql")
        return sparql.strip() if sparql else None

    if "sparql" in item and item["sparql"]:
        return str(item["sparql"]).strip()

    return None


def call_generation_api(
    api_url: str,
    question_text: str,
    dataset_url: str,
    model_name: str = "openai/gpt-4o-mini",
    log_calls: bool = False,
    use_translate: bool = True,
    use_llm_translate: bool = False,
    timeout: int = 600,
) -> Tuple[Optional[str], Optional[str], Optional[str], int, int, int, List[str]]:
    try:
        response = requests.get(
            api_url,
            params={
                "question": question_text,
                "dataset": dataset_url,
                "model_name": model_name,
                "log_calls": log_calls,
                "use_translate": use_translate,
                "use_llm_translate": use_llm_translate,
            },
            timeout=timeout,
        )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text}", None, 0, 0, 0, []

        api_data = response.json()
        generated_query = api_data.get("query")
        translated_question = api_data.get("translated_question")
        prompt_tokens = api_data.get("prompt_tokens", 0)
        completion_tokens = api_data.get("completion_tokens", 0)
        req_count = api_data.get("requests", 0)
        step_times = api_data.get("step_times", [])

        if not generated_query:
            return None, "API returned empty query", translated_question, prompt_tokens, completion_tokens, req_count, step_times

        return str(generated_query).strip(), None, translated_question, prompt_tokens, completion_tokens, req_count, step_times

    except requests.exceptions.Timeout:
        raise  # propagate so callers can detect a dead port
    except Exception as e:
        return None, str(e), None, 0, 0, 0, []


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
                "User-Agent": "text2sparql-evaluation/1.0 (https://github.com/WuytsW/text2sparql)",
            },
            timeout=timeout,
        )

        if response.status_code == 200:
            return response.json(), None

        return None, f"HTTP {response.status_code}: {response.text}"

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
        ordered_values = [
            normalize_binding_value(row[var_name])
            for var_name in sorted(row.keys())
        ]
        if not ordered_values:
            continue
        if len(ordered_values) == 1:
            values.add(ordered_values[0])
        else:
            values.add(" | ".join(ordered_values))

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
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return tp, fp, fn, precision, recall, f1


def compute_micro_scores(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_macro_scores(precisions: List[float], recalls: List[float], f1s: List[float]) -> Tuple[float, float, float]:
    n = len(f1s)
    if n == 0:
        return 0.0, 0.0, 0.0
    return sum(precisions) / n, sum(recalls) / n, sum(f1s) / n


def make_summary(
    total_questions: int,
    processed_questions: int,
    skipped_questions: int,
    running_tp: int,
    running_fp: int,
    running_fn: int,
    per_question_precisions: List[float],
    per_question_recalls: List[float],
    per_question_f1s: List[float],
    errors_count: int,
    started_at: float,
    total_prompt_tokens: int = 0,
    total_completion_tokens: int = 0,
    total_requests: int = 0,
    total_cost: float = 0.0,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    micro_p, micro_r, micro_f1 = compute_micro_scores(running_tp, running_fp, running_fn)
    macro_p, macro_r, macro_f1 = compute_macro_scores(per_question_precisions, per_question_recalls, per_question_f1s)

    summary: Dict[str, Any] = {
        "total_questions": total_questions,
        "processed_questions": processed_questions,
        "skipped_questions": skipped_questions,
        "remaining_questions": total_questions - processed_questions - skipped_questions,
        "evaluated_questions": processed_questions,
        "final_tp_so_far": running_tp,
        "final_fp_so_far": running_fp,
        "final_fn_so_far": running_fn,
        "micro_precision_so_far": micro_p,
        "micro_recall_so_far": micro_r,
        "micro_f1_so_far": micro_f1,
        "macro_precision_so_far": macro_p,
        "macro_recall_so_far": macro_r,
        "macro_f1_so_far": macro_f1,
        "questions_with_f1_less_than_1": errors_count,
        "elapsed_seconds": time.time() - started_at,
        "done": (processed_questions + skipped_questions) == total_questions,
        "prompt_tokens_so_far": total_prompt_tokens,
        "completion_tokens_so_far": total_completion_tokens,
        "total_requests_so_far": total_requests,
        "total_cost_so_far": total_cost,
    }

    if lang is not None:
        summary["language"] = lang

    return summary


# ─── Per-language state ──────────────────────────────────────────────────────

class LangState:
    def __init__(self, lang: str, lang_dir: Path, total_questions: int, started_at: float) -> None:
        self.lang = lang
        self.lang_dir = lang_dir
        self.started_at = started_at
        self.total_questions = total_questions

        self.results: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.skipped: List[Dict[str, Any]] = []

        self.running_tp = 0
        self.running_fp = 0
        self.running_fn = 0
        self.processed = 0
        self.skipped_count = 0

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
        self.total_cost = 0.0

        self.per_q_precisions: List[float] = []
        self.per_q_recalls: List[float] = []
        self.per_q_f1s: List[float] = []

        lang_dir.mkdir(parents=True, exist_ok=True)
        flush_json(lang_dir / "results.json", [])
        flush_json(lang_dir / "errors.json", [])
        flush_json(lang_dir / "skipped_empty_gold.json", [])
        flush_json(
            lang_dir / "summary.json",
            make_summary(
                total_questions=total_questions,
                processed_questions=0, skipped_questions=0,
                running_tp=0, running_fp=0, running_fn=0,
                per_question_precisions=[], per_question_recalls=[], per_question_f1s=[],
                errors_count=0, started_at=started_at, lang=lang,
            ),
        )

    def flush_summary(self) -> None:
        flush_json(
            self.lang_dir / "summary.json",
            make_summary(
                total_questions=self.total_questions,
                processed_questions=self.processed,
                skipped_questions=self.skipped_count,
                running_tp=self.running_tp, running_fp=self.running_fp, running_fn=self.running_fn,
                per_question_precisions=self.per_q_precisions,
                per_question_recalls=self.per_q_recalls,
                per_question_f1s=self.per_q_f1s,
                errors_count=len(self.errors),
                started_at=self.started_at,
                total_prompt_tokens=self.total_prompt_tokens,
                total_completion_tokens=self.total_completion_tokens,
                total_requests=self.total_requests,
                total_cost=self.total_cost,
                lang=self.lang,
            ),
        )

    def final_summary(self) -> Dict[str, Any]:
        return make_summary(
            total_questions=self.total_questions,
            processed_questions=self.processed,
            skipped_questions=self.skipped_count,
            running_tp=self.running_tp, running_fp=self.running_fp, running_fn=self.running_fn,
            per_question_precisions=self.per_q_precisions,
            per_question_recalls=self.per_q_recalls,
            per_question_f1s=self.per_q_f1s,
            errors_count=len(self.errors),
            started_at=self.started_at,
            total_prompt_tokens=self.total_prompt_tokens,
            total_completion_tokens=self.total_completion_tokens,
            total_requests=self.total_requests,
            total_cost=self.total_cost,
            lang=self.lang,
        )


# ─── Main evaluation ─────────────────────────────────────────────────────────

def evaluate_all_languages(
    input_file: Path,
    output_dir: Path,
    api_urls: List[str],
    sparql_endpoint: str,
    languages: Optional[List[str]],
    limit: Optional[int],
    sleep_seconds: float,
    model_name: str = "openai/gpt-4o-mini",
    log_calls: bool = False,
    use_translate: bool = True,
    use_llm_translate: bool = False,
    cost_prompt: float = 0.15,
    cost_completion: float = 0.60,
) -> None:
    questions_data = load_questions(input_file)

    if languages:
        langs = languages
    else:
        langs = detect_languages(questions_data)
        logging.info("Auto-detected languages: %s", langs)

    if not langs:
        logging.error("No languages found in dataset.")
        sys.exit(1)

    if limit is not None:
        questions_data = questions_data[:limit]

    total_questions = len(questions_data)
    print(f"\nEvaluating {len(langs)} language(s): {', '.join(langs)}")
    print(f"Questions: {total_questions}\n")

    started_at = time.time()

    # Count how many questions each language actually has (for summary totals)
    lang_q_count = {
        lang: sum(1 for q in questions_data if has_language(q.get("question"), lang))
        for lang in langs
    }

    states: Dict[str, LangState] = {
        lang: LangState(
            lang=lang,
            lang_dir=output_dir / lang,
            total_questions=lang_q_count[lang],
            started_at=started_at,
        )
        for lang in langs
    }

    logging.info("Using %d API instance(s): %s", len(api_urls), api_urls)

    # ── Phase A: pre-compute gold results and enqueue all active work items ───

    work_queue: queue.Queue = queue.Queue()
    gold_cache: Dict[int, Tuple[Optional[Dict[str, Any]], Optional[str], Set[str]]] = {}

    for q_idx, item in enumerate(questions_data, start=1):
        qid = item.get("id", q_idx)
        gold_query = extract_gold_sparql(item)

        if q_idx not in gold_cache:
            if gold_query:
                gold_result, gold_error = execute_sparql(gold_query, sparql_endpoint)
                gold_values = extract_values(gold_result)
            else:
                gold_result, gold_error, gold_values = None, "No gold query provided", set()
            gold_cache[q_idx] = (gold_result, gold_error, gold_values)

        gold_result, gold_error, gold_values = gold_cache[q_idx]
        question_field = item.get("question")

        for lang in langs:
            if not has_language(question_field, lang):
                continue
            state = states[lang]
            question_text = extract_question(question_field, lang=lang)
            logging.info("[%s] (%d/%d) QID=%s | %s", lang, q_idx, total_questions, qid, question_text)

            if has_empty_bindings(gold_result):
                state.skipped_count += 1
                state.skipped.append({
                    "index": q_idx,
                    "id": qid,
                    "lang": lang,
                    "question": question_text,
                    "gold_query": gold_query,
                    "gold_result": gold_result,
                    "gold_execution_error": gold_error,
                    "reason": "Skipped because gold query returned empty result",
                    "gold_values": sorted(gold_values),
                })
                flush_json(state.lang_dir / "skipped_empty_gold.json", state.skipped)
                state.flush_summary()
                logging.info("[%s] QID=%s skipped | gold result empty", lang, qid)
                print(f"  [{lang}] Q{q_idx} skipped (empty gold)")
            else:
                work_queue.put((q_idx, lang, question_text, gold_query, gold_values, gold_error, qid))

    logging.info(
        "Gold pre-computation done. %d work item(s) queued across %d port(s).",
        work_queue.qsize(), len(api_urls),
    )

    # ── Phase B: parallel API calls — one worker thread per port ─────────────

    results_lock = threading.Lock()

    def port_worker(url: str) -> None:
        while True:
            try:
                work_item = work_queue.get(block=True, timeout=2.0)
            except queue.Empty:
                return

            q_idx, lang, question_text, gold_query, gold_values, gold_error, qid = work_item

            timed_out = False
            task = None
            try:
                task = _call_api_and_execute(
                    api_url=url,
                    question_text=question_text,
                    sparql_endpoint=sparql_endpoint,
                    model_name=model_name,
                    log_calls=log_calls,
                    use_translate=use_translate,
                    use_llm_translate=use_llm_translate,
                )
            except requests.exceptions.Timeout:
                timed_out = True

            if timed_out:
                logging.warning(
                    "Port %s timed out — marking as dead, requeueing [%s] Q%d (QID=%s)",
                    url, lang, q_idx, qid,
                )
                work_queue.put(work_item)  # let another port pick it up
                work_queue.task_done()
                return  # this port is dead; stop using it

            generated_query = task["generated_query"]
            generated_error = task["generated_error"]
            translated_question = task["translated_question"]
            api_prompt_tokens = task["api_prompt_tokens"]
            api_completion_tokens = task["api_completion_tokens"]
            api_requests = task["api_requests"]
            step_times = task["step_times"]
            generated_values: Set[str] = task["generated_values"]

            api_cost = (
                api_prompt_tokens * cost_prompt / 1_000_000
                + api_completion_tokens * cost_completion / 1_000_000
            )
            tp, fp, fn, precision, recall, f1 = compute_metrics(generated_values, gold_values)

            with results_lock:
                state = states[lang]
                state.running_tp += tp
                state.running_fp += fp
                state.running_fn += fn
                state.processed += 1
                state.total_prompt_tokens += api_prompt_tokens
                state.total_completion_tokens += api_completion_tokens
                state.total_requests += api_requests
                state.total_cost += api_cost
                state.per_q_precisions.append(precision)
                state.per_q_recalls.append(recall)
                state.per_q_f1s.append(f1)

                micro_p, micro_r, micro_f1 = compute_micro_scores(
                    state.running_tp, state.running_fp, state.running_fn
                )
                macro_p, macro_r, macro_f1 = compute_macro_scores(
                    state.per_q_precisions, state.per_q_recalls, state.per_q_f1s
                )

                logging.info(
                    "[%s] QID=%s done | P=%.4f R=%.4f F1=%.4f | gold=%d pred=%d | port=%s",
                    lang, qid, precision, recall, f1,
                    len(gold_values), len(generated_values), url,
                )

                result_entry = {
                    "index": q_idx,
                    "id": qid,
                    "lang": lang,
                    "question": question_text,
                    "translated_question": translated_question,
                    "generated_query": generated_query,
                    "generated_execution_error": generated_error,
                    "gold_query": gold_query,
                    "gold_execution_error": gold_error,
                    "f1": f1,
                    "running_micro_f1": micro_f1,
                    "running_macro_f1": macro_f1,
                    "prompt_tokens": api_prompt_tokens,
                    "completion_tokens": api_completion_tokens,
                    "requests": api_requests,
                    "cost": api_cost,
                    "step_times": step_times,
                    "generated_values": sorted(generated_values),
                    "gold_values": sorted(gold_values),
                    "only_in_generated": sorted(generated_values - gold_values),
                    "only_in_gold": sorted(gold_values - generated_values),
                }

                state.results.append(result_entry)
                state.results.sort(key=lambda r: r["index"])
                flush_json(state.lang_dir / "results.json", state.results)

                if f1 < 1.0:
                    state.errors.append(result_entry)
                    state.errors.sort(key=lambda r: r["index"])
                    flush_json(state.lang_dir / "errors.json", state.errors)

                state.flush_summary()

                print(f"  [{lang}] Q{q_idx}: {question_text}")
                if translated_question:
                    print(f"         -> {translated_question}")
                print(f"         F1={f1:.4f}  micro={micro_f1:.4f}  macro=\033[33m{macro_f1:.4f}\033[0m")

            work_queue.task_done()

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    threads = [
        threading.Thread(target=port_worker, args=(url,), daemon=True, name=f"worker-{url}")
        for url in api_urls
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Report work items that could not be processed because all ports died
    unprocessed: List[Tuple] = []
    while not work_queue.empty():
        try:
            unprocessed.append(work_queue.get_nowait())
        except queue.Empty:
            break
    if unprocessed:
        logging.error(
            "%d work item(s) unprocessed — all ports became unavailable: %s",
            len(unprocessed),
            [(u[0], u[1]) for u in unprocessed],  # (q_idx, lang) pairs
        )

    # Build combined summary
    per_lang_summaries = {lang: states[lang].final_summary() for lang in langs}

    combined_tp = sum(s["final_tp_so_far"] for s in per_lang_summaries.values())
    combined_fp = sum(s["final_fp_so_far"] for s in per_lang_summaries.values())
    combined_fn = sum(s["final_fn_so_far"] for s in per_lang_summaries.values())
    combined_micro_p, combined_micro_r, combined_micro_f1 = compute_micro_scores(combined_tp, combined_fp, combined_fn)

    avg_macro_p, avg_macro_r, avg_macro_f1 = compute_macro_scores(
        [s["macro_precision_so_far"] for s in per_lang_summaries.values()],
        [s["macro_recall_so_far"] for s in per_lang_summaries.values()],
        [s["macro_f1_so_far"] for s in per_lang_summaries.values()],
    )

    combined_summary = {
        "languages": langs,
        "elapsed_seconds": time.time() - started_at,
        "combined_tp": combined_tp,
        "combined_fp": combined_fp,
        "combined_fn": combined_fn,
        "combined_micro_precision": combined_micro_p,
        "combined_micro_recall": combined_micro_r,
        "combined_micro_f1": combined_micro_f1,
        "avg_macro_precision": avg_macro_p,
        "avg_macro_recall": avg_macro_r,
        "avg_macro_f1": avg_macro_f1,
        "micro_f1_per_language": {lang: s["micro_f1_so_far"] for lang, s in per_lang_summaries.items()},
        "macro_f1_per_language": {lang: s["macro_f1_so_far"] for lang, s in per_lang_summaries.items()},
        "total_prompt_tokens": sum(s["prompt_tokens_so_far"] for s in per_lang_summaries.values()),
        "total_completion_tokens": sum(s["completion_tokens_so_far"] for s in per_lang_summaries.values()),
        "total_requests": sum(s["total_requests_so_far"] for s in per_lang_summaries.values()),
        "total_cost": sum(s["total_cost_so_far"] for s in per_lang_summaries.values()),
        "per_language": per_lang_summaries,
    }

    flush_json(output_dir / "summary_all_languages.json", combined_summary)

    print("\n\n" + "=" * 60)
    print("  RESULTS PER LANGUAGE")
    print("=" * 60)
    for lang in langs:
        s = per_lang_summaries[lang]
        print(
            f"  [{lang:4s}]  micro F1: {s['micro_f1_so_far']:.4f}  "
            f"macro F1: {s['macro_f1_so_far']:.4f}  "
            f"({s['evaluated_questions']} evaluated, {s['skipped_questions']} skipped)"
        )
    print(f"\n  Combined micro F1 : {combined_micro_f1:.4f}")
    print(f"  Avg    macro F1   : {avg_macro_f1:.4f}")
    print(f"  Total cost        : ${combined_summary['total_cost']:.4f}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QALD evaluator — all languages, question-interleaved order"
    )
    parser.add_argument("--input-file", type=Path, help="Path to QALD JSON file", default="qald_9_plus_test_dbpedia.json")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL)
    parser.add_argument(
        "--num-ports", type=int, default=1, metavar="N",
        help="Number of API instances to use. Ports are assigned starting from the "
             "--api-url port (e.g. --num-ports 5 uses 8000-8004). Default: 1",
    )
    parser.add_argument(
        "--low-port", type=int, default=None, metavar="PORT",
        help="Lowest port number to use (overrides the port in --api-url). Default: port from --api-url",
    )
    parser.add_argument("--sparql-endpoint", type=str, default=DEFAULT_SPARQL_ENDPOINT)
    parser.add_argument(
        "--languages", type=str, default=None,
        help="Comma-separated languages to evaluate (default: auto-detect from dataset)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max questions to evaluate")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--model-name", type=str, default="openai/gpt-4o-mini")
    parser.add_argument("--test-name", type=str, default="All languages", help="Label appended to output folder")
    parser.add_argument("--log-calls", action="store_true", default=True)
    parser.add_argument("--cost-prompt", type=float, default=0.15, help="Cost per 1M prompt tokens (USD)")
    parser.add_argument("--cost-completion", type=float, default=0.60, help="Cost per 1M completion tokens (USD)")
    parser.add_argument("--no-translate", action="store_true", default=False,
                        help="Disable translation step (sends use_translate=False to the API)")
    parser.add_argument("--use-llm-translate", action="store_true", default=True,
                        help="Use LLM-based translation instead of standard translation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    languages: Optional[List[str]] = None
    if args.languages:
        languages = [l.strip() for l in args.languages.split(",") if l.strip()]

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    lang_tag = args.languages.replace(",", "-") if args.languages else "all-languages"
    folder_name = (
        f"evaluation_{Path(args.input_file).stem}_{timestamp}"
        f"_{lang_tag}_{args.model_name.replace('/', '-').replace(':', '-')}"
    )
    if args.test_name:
        folder_name += f"_{args.test_name}"
    output_dir = Path("test_results") / folder_name

    setup_logging(output_dir)

    api_urls = build_api_urls(args.api_url, args.num_ports, low_port=args.low_port)
    if len(api_urls) > 1:
        logging.info("Distributing across %d API instances: %s", len(api_urls), api_urls)

    evaluate_all_languages(
        input_file=args.input_file,
        output_dir=output_dir,
        api_urls=api_urls,
        sparql_endpoint=args.sparql_endpoint,
        languages=languages,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
        model_name=args.model_name,
        log_calls=args.log_calls,
        use_translate=not args.no_translate,
        use_llm_translate=args.use_llm_translate,
        cost_prompt=args.cost_prompt,
        cost_completion=args.cost_completion,
    )


if __name__ == "__main__":
    main()
