#!/usr/bin/env python3
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
    """Return list of API URLs for ports low_port .. low_port+num_ports-1.

    If *low_port* is given it overrides whatever port is in *base_url*.
    """
    parsed = urlparse(base_url)
    base_port = low_port if low_port is not None else (parsed.port or 8000)
    hostname = parsed.hostname or "localhost"
    return [
        urlunparse(parsed._replace(netloc=f"{hostname}:{base_port + i}"))
        for i in range(num_ports)
    ]

# Cost per token (adjust to match the model being used)
COST_PER_PROMPT_TOKEN: float = 0.15 / 1_000_000
COST_PER_COMPLETION_TOKEN: float = 0.60 / 1_000_000

# Retry settings for transient SPARQL endpoint errors
SPARQL_RETRY_STATUS_CODES: set = {429, 500, 502, 503, 504}
SPARQL_MAX_RETRIES: int = 3          # number of retries (not counting first attempt)
SPARQL_RETRY_BASE_DELAY: float = 5.0 # seconds; doubles each retry (5 → 10 → 20)


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


def call_generation_api(
    api_url: str,
    question_text: str,
    dataset_url: str,
    model_name: str = "openai/gpt-4o-mini",
    log_calls: bool = False,
    temperature: float = 0.0,
    use_translate: bool = True,
    use_llm_translate: bool = False,
    use_icl: bool = True,
    use_eat: bool = True,
    use_context: bool = True,
    timeout: int = 600,
) -> Tuple[Optional[str], Optional[str], Optional[str], int, int, int, List[str]]:
    try:
        response = requests.get(
            api_url,
            params={"question": question_text, "dataset": dataset_url, "model_name": model_name, "log_calls": log_calls, "temperature": temperature, "use_translate": use_translate, "use_llm_translate": use_llm_translate, "use_icl": use_icl, "use_eat": use_eat, "use_context": use_context},
            timeout=timeout,
        )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text}", None, 0, 0, 0, []

        api_data = response.json()
        logging.debug("API response: %s", api_data)
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

    last_error: Optional[str] = None

    for attempt in range(SPARQL_MAX_RETRIES + 1):
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

            last_error = f"HTTP {response.status_code}: {response.text}"

            if response.status_code in SPARQL_RETRY_STATUS_CODES and attempt < SPARQL_MAX_RETRIES:
                delay = SPARQL_RETRY_BASE_DELAY * (2 ** attempt)
                logging.warning(
                    "SPARQL endpoint returned %s (attempt %d/%d). Retrying in %.0fs…",
                    response.status_code, attempt + 1, SPARQL_MAX_RETRIES + 1, delay,
                )
                time.sleep(delay)
                continue

            return None, last_error

        except Exception as e:
            last_error = str(e)
            if attempt < SPARQL_MAX_RETRIES:
                delay = SPARQL_RETRY_BASE_DELAY * (2 ** attempt)
                logging.warning(
                    "SPARQL request exception (attempt %d/%d): %s. Retrying in %.0fs…",
                    attempt + 1, SPARQL_MAX_RETRIES + 1, last_error, delay,
                )
                time.sleep(delay)
            else:
                return None, last_error

    return None, last_error


def canonicalize_uri(value: str) -> str:
    value = value.strip()

    doubled_prefix = "http://dbpedia.org/resource/http://dbpedia.org/resource/"
    if value.startswith(doubled_prefix):
        value = value.replace(doubled_prefix, "http://dbpedia.org/resource/", 1)

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
    """
    Extract comparison values from a SPARQL JSON result.

    - For boolean ASK results, compare the boolean value.
    - For single-column SELECT results, compare just the normalized value.
    - For multi-column SELECT results, compare tuples of normalized values
      in sorted variable-name order, without including the variable names.
    """
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
    """
    Return True when a SELECT result has no bindings.
    Boolean ASK results are not treated as empty.
    """
    if not result_json:
        return True

    if "boolean" in result_json:
        return False

    bindings = result_json.get("results", {}).get("bindings", [])
    return len(bindings) == 0


def compute_metrics(predicted_set: Set[str], gold_set: Set[str]) -> Tuple[int, int, int, float, float, float]:
    tp = len(predicted_set & gold_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)

    if not predicted_set and not gold_set:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
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
) -> Dict[str, Any]:
    micro_precision, micro_recall, micro_f1 = compute_micro_scores(running_tp, running_fp, running_fn)
    macro_precision, macro_recall, macro_f1 = compute_macro_scores(
        per_question_precisions, per_question_recalls, per_question_f1s
    )

    return {
        "total_questions": total_questions,
        "processed_questions": processed_questions,
        "skipped_questions": skipped_questions,
        "remaining_questions": total_questions - processed_questions - skipped_questions,
        "evaluated_questions": processed_questions,
        "final_tp_so_far": running_tp,
        "final_fp_so_far": running_fp,
        "final_fn_so_far": running_fn,
        "micro_precision_so_far": micro_precision,
        "micro_recall_so_far": micro_recall,
        "micro_f1_so_far": micro_f1,
        "macro_precision_so_far": macro_precision,
        "macro_recall_so_far": macro_recall,
        "macro_f1_so_far": macro_f1,
        "questions_with_f1_less_than_1": errors_count,
        "elapsed_seconds": time.time() - started_at,
        "done": (processed_questions + skipped_questions) == total_questions,
        "prompt_tokens_so_far": total_prompt_tokens,
        "completion_tokens_so_far": total_completion_tokens,
        "total_requests_so_far": total_requests,
        "total_cost_so_far": total_cost,
    }



















def evaluate(
    input_file: Path,
    output_dir: Path,
    api_url: str,
    sparql_endpoint: str,
    lang: str,
    limit: Optional[int],
    sleep_seconds: float,
    num_ports: int = 1,
    low_port: Optional[int] = None,
    model_name: str = "openai/gpt-4o-mini",
    log_calls: bool = False,
    temperature: float = 0.0,
    use_translate: bool = True,
    use_llm_translate: bool = False,
    use_icl: bool = True,
    use_eat: bool = True,
    use_context: bool = True,
    cost_prompt: float = 0.15,
    cost_completion: float = 0.60,
    start_at_index: int = 0,
    end_at_index: Optional[int] = None,
    resume: bool = False,
    rerun_from_question: Optional[int] = None,
) -> None:
    questions_data_full = load_questions(input_file)
    api_urls = build_api_urls(api_url, num_ports, low_port)
    total_questions = len(questions_data_full)

    output_file = output_dir / "results.json"
    error_file = output_dir / "errors.json"
    skipped_file = output_dir / "skipped_empty_gold.json"
    summary_file = output_dir / "summary.json"

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    running_tp = 0
    running_fp = 0
    running_fn = 0

    processed_questions = 0
    skipped_questions = 0

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_requests = 0
    total_cost = 0.0

    per_question_precisions: List[float] = []
    per_question_recalls: List[float] = []
    per_question_f1s: List[float] = []

    started_at = time.time()

    if resume:
        results = json.loads(output_file.read_text(encoding="utf-8"))
        errors = json.loads(error_file.read_text(encoding="utf-8")) if error_file.exists() else []
        skipped = json.loads(skipped_file.read_text(encoding="utf-8")) if skipped_file.exists() else []

        if rerun_from_question is not None:
            results = [r for r in results if r.get("index", 0) < rerun_from_question]
            skipped = [s for s in skipped if s.get("index", 0) < rerun_from_question]
            errors  = [e for e in errors  if e.get("index", 0) < rerun_from_question]
            flush_json(output_file,  results)
            flush_json(error_file,   errors)
            flush_json(skipped_file, skipped)
            logging.info(
                "Rolled back to question %d: keeping %d results, %d skipped",
                rerun_from_question, len(results), len(skipped)
            )

        for r in results:
            gen_vals = set(r.get("generated_values", []))
            gold_vals = set(r.get("gold_values", []))
            tp, fp, fn, p, rec, f1 = compute_metrics(gen_vals, gold_vals)
            running_tp += tp
            running_fp += fp
            running_fn += fn
            per_question_precisions.append(p)
            per_question_recalls.append(rec)
            per_question_f1s.append(f1)

        processed_questions = len(results)
        skipped_questions = len(skipped)
        start_at_index = processed_questions + skipped_questions

        if summary_file.exists():
            summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
            total_prompt_tokens = summary_data.get("prompt_tokens_so_far", 0)
            total_completion_tokens = summary_data.get("completion_tokens_so_far", 0)
            total_requests = summary_data.get("total_requests_so_far", 0)
            total_cost = summary_data.get("total_cost_so_far", 0.0)

        if rerun_from_question is not None:
            # Summary totals include the discarded questions — recompute from the kept results.
            total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in results)
            total_completion_tokens = sum(r.get("completion_tokens", 0) for r in results)
            total_requests = sum(r.get("requests", 0) for r in results)
            total_cost = sum(r.get("cost", 0.0) for r in results)

        logging.info("Resuming from index %d (%d processed, %d skipped)", start_at_index, processed_questions, skipped_questions)

    questions_data = questions_data_full[start_at_index:end_at_index]
    if limit is not None:
        questions_data = questions_data[:limit]

    results_lock = threading.Lock()
    question_queue: queue.Queue = queue.Queue()

    # ── Phase A: execute gold queries (sequential, fast) and enqueue active ──

    if not resume:
        flush_json(output_file, results)
        flush_json(error_file, errors)
        flush_json(skipped_file, skipped)
        flush_json(
            summary_file,
            make_summary(
                total_questions=total_questions,
                processed_questions=0,
                skipped_questions=0,
                running_tp=0,
                running_fp=0,
                running_fn=0,
                per_question_precisions=[],
                per_question_recalls=[],
                per_question_f1s=[],
                errors_count=0,
                started_at=started_at,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_requests=0,
                total_cost=0.0,
            ),
        )

    for i, item in enumerate(questions_data, start=start_at_index + 1):
        question_text = extract_question(item.get("question"), lang=lang)
        gold_query = extract_gold_sparql(item)
        qid = item.get("id", i)

        logging.info("(%d/%d) QID=%s | gold pre-compute | %s", i, total_questions, qid, question_text)

        gold_result = None
        gold_error = None
        gold_values: Set[str] = set()

        if gold_query:
            gold_result, gold_error = execute_sparql(gold_query, sparql_endpoint)
            gold_values = extract_values(gold_result)
        else:
            gold_error = "No gold query provided"

        if has_empty_bindings(gold_result):
            skipped_questions += 1

            skipped_entry = {
                "index": i,
                "id": qid,
                "question": question_text,
                "gold_query": gold_query,
                "gold_result": gold_result,
                "gold_execution_error": gold_error,
                "reason": "Skipped because gold query returned empty result",
                "gold_values": sorted(gold_values),
            }
            skipped.append(skipped_entry)
            flush_json(skipped_file, skipped)

            summary = make_summary(
                total_questions=total_questions,
                processed_questions=processed_questions,
                skipped_questions=skipped_questions,
                running_tp=running_tp,
                running_fp=running_fp,
                running_fn=running_fn,
                per_question_precisions=per_question_precisions,
                per_question_recalls=per_question_recalls,
                per_question_f1s=per_question_f1s,
                errors_count=len(errors),
                started_at=started_at,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_requests=total_requests,
                total_cost=total_cost,
            )
            flush_json(summary_file, summary)

            logging.info(
                "QID=%s skipped | gold result empty%s",
                qid,
                f" | error: {gold_error}" if gold_error else "",
            )
            print(f"\n[{i}/{total_questions}] {question_text}")
            print("Skipped: gold query returned empty result")
        else:
            question_queue.put((i, qid, question_text, gold_query, gold_values, gold_error))

    # ── Phase B: parallel API calls — one worker thread per port ─────────────

    logging.info(
        "Gold pre-computation done. Starting parallel evaluation with %d port(s): %s",
        len(api_urls), api_urls,
    )

    def worker(api_url: str) -> None:
        nonlocal running_tp, running_fp, running_fn
        nonlocal processed_questions
        nonlocal total_prompt_tokens, total_completion_tokens, total_requests, total_cost

        while True:
            try:
                work_item = question_queue.get(block=True, timeout=2.0)
            except queue.Empty:
                return

            i, qid, question_text, gold_query, gold_values, gold_error = work_item

            generated_query: Optional[str] = None
            generated_result = None
            generated_error: Optional[str] = None
            generated_values: Set[str] = set()
            api_prompt_tokens = 0
            api_completion_tokens = 0
            api_requests = 0
            translated_question: Optional[str] = None
            step_times: List[str] = []

            timed_out = False
            try:
                generated_query, api_error, translated_question, \
                    api_prompt_tokens, api_completion_tokens, api_requests, step_times = call_generation_api(
                        api_url=api_url,
                        question_text=question_text,
                        dataset_url=sparql_endpoint,
                        model_name=model_name,
                        log_calls=log_calls,
                        temperature=temperature,
                        use_translate=use_translate,
                        use_llm_translate=use_llm_translate,
                        use_icl=use_icl,
                        use_eat=use_eat,
                        use_context=use_context,
                    )
                generated_error = api_error

                if generated_query:
                    generated_result, exec_error = execute_sparql(generated_query, sparql_endpoint)
                    if exec_error:
                        generated_error = exec_error

                generated_values = extract_values(generated_result)

            except requests.exceptions.Timeout:
                timed_out = True
            except Exception as e:
                generated_error = str(e)

            if timed_out:
                logging.warning(
                    "Port %s timed out — marking as dead, requeueing Q%d (QID=%s)",
                    api_url, i, qid,
                )
                question_queue.put(work_item)  # let another port pick it up
                question_queue.task_done()
                return  # this port is dead; stop using it

            api_cost = (
                api_prompt_tokens * cost_prompt / 1_000_000
                + api_completion_tokens * cost_completion / 1_000_000
            )
            tp, fp, fn, precision, recall, f1 = compute_metrics(generated_values, gold_values)

            with results_lock:
                running_tp += tp
                running_fp += fp
                running_fn += fn
                processed_questions += 1
                total_prompt_tokens += api_prompt_tokens
                total_completion_tokens += api_completion_tokens
                total_requests += api_requests
                total_cost += api_cost

                per_question_precisions.append(precision)
                per_question_recalls.append(recall)
                per_question_f1s.append(f1)

                r_micro_p, r_micro_r, r_micro_f1 = compute_micro_scores(
                    running_tp, running_fp, running_fn
                )
                r_macro_p, r_macro_r, r_macro_f1 = compute_macro_scores(
                    per_question_precisions, per_question_recalls, per_question_f1s
                )

                logging.info(
                    "QID=%s done | P=%.4f R=%.4f F1=%.4f | gold=%d pred=%d | port=%s",
                    qid, precision, recall, f1,
                    len(gold_values), len(generated_values), api_url,
                )

                result_entry = {
                    "index": i,
                    "id": qid,
                    "question": question_text,
                    "translated_question": translated_question,
                    "generated_query": generated_query,
                    "generated_execution_error": generated_error,
                    "gold_query": gold_query,
                    "gold_execution_error": gold_error,
                    "f1": f1,
                    "running_micro_f1": r_micro_f1,
                    "running_macro_f1": r_macro_f1,
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

                results.append(result_entry)
                results.sort(key=lambda r: r["index"])
                flush_json(output_file, results)

                if f1 < 1.0:
                    errors.append(result_entry)
                    errors.sort(key=lambda r: r["index"])
                    flush_json(error_file, errors)

                summary = make_summary(
                    total_questions=total_questions,
                    processed_questions=processed_questions,
                    skipped_questions=skipped_questions,
                    running_tp=running_tp,
                    running_fp=running_fp,
                    running_fn=running_fn,
                    per_question_precisions=per_question_precisions,
                    per_question_recalls=per_question_recalls,
                    per_question_f1s=per_question_f1s,
                    errors_count=len(errors),
                    started_at=started_at,
                    total_prompt_tokens=total_prompt_tokens,
                    total_completion_tokens=total_completion_tokens,
                    total_requests=total_requests,
                    total_cost=total_cost,
                )
                flush_json(summary_file, summary)

                print(f"\n[{i}/{total_questions}] {question_text}")
                if translated_question:
                    print(f"Translated question: {translated_question}")
                print(f"Question P/R/F1: {precision:.4f} / {recall:.4f} / \033[33m{f1:.4f}\033[0m")
                print(f"Running micro P/R/F1: {r_micro_p:.4f} / {r_micro_r:.4f} / {r_micro_f1:.4f}")
                print(f"Running macro P/R/F1: {r_macro_p:.4f} / {r_macro_r:.4f} / \033[33m{r_macro_f1:.4f}\033[0m\n")

            question_queue.task_done()

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    threads = [
        threading.Thread(target=worker, args=(url,), daemon=True, name=f"worker-{url}")
        for url in api_urls
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Report questions that could not be processed because all ports died
    unprocessed_indices: List[int] = []
    while not question_queue.empty():
        try:
            item = question_queue.get_nowait()
            unprocessed_indices.append(item[0])
        except queue.Empty:
            break
    if unprocessed_indices:
        logging.error(
            "%d question(s) unprocessed — all ports became unavailable: indices %s",
            len(unprocessed_indices), unprocessed_indices,
        )

    final_summary = make_summary(
        total_questions=total_questions,
        processed_questions=processed_questions,
        skipped_questions=skipped_questions,
        running_tp=running_tp,
        running_fp=running_fp,
        running_fn=running_fn,
        per_question_precisions=per_question_precisions,
        per_question_recalls=per_question_recalls,
        per_question_f1s=per_question_f1s,
        errors_count=len(errors),
        started_at=started_at,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_requests=total_requests,
        total_cost=total_cost,
    )
    flush_json(summary_file, final_summary)

    print("\nFinished processing all questions")
    print(json.dumps(final_summary, indent=4, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live QALD evaluator for DBpedia")
    parser.add_argument("--input-file", type=Path, required=True, help="Path to QALD JSON file")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="Your FastAPI /api endpoint")
    parser.add_argument(
        "--num-ports", type=int, default=1, metavar="N",
        help="Number of parallel API instances. Ports are base_port .. base_port+N-1. Default: 1",
    )
    parser.add_argument(
        "--low-port", type=int, default=None, metavar="PORT",
        help="Lowest port number to use (overrides the port in --api-url). Default: port from --api-url",
    )
    parser.add_argument("--sparql-endpoint", type=str, default=DEFAULT_SPARQL_ENDPOINT, help="SPARQL endpoint to execute against")
    parser.add_argument("--lang", type=str, default="en", help="Question language")
    parser.add_argument("--limit", type=int, default=None, help="Optional question limit")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between questions")
    parser.add_argument("--agent-log", action="store_true", help="Whether to log LLM calls to console and file")
    parser.add_argument("--model-name", type=str, default="openai/gpt-4o-mini", help="OpenRouter model identifier")
    parser.add_argument("--test-name", type=str, default=None, help="Optional name appended to the output folder")
    parser.add_argument("--log-calls", action="store_true", help="Whether to log API calls and SPARQL executions in detail")
    parser.add_argument("--cost-prompt", type=float, default=0.15, help="Cost per 1M prompt tokens in USD (default: 0.15)")
    parser.add_argument("--cost-completion", type=float, default=0.60, help="Cost per 1M completion tokens in USD (default: 0.60)")
    parser.add_argument("--no-translate", action="store_true", help="Disable question translation")
    parser.add_argument("--use-llm-translate", action="store_true", help="Use LLM-based translation instead of standard translation")
    parser.add_argument("--no-icl", action="store_true", help="Disable in-context learning examples")
    parser.add_argument("--no-eat", action="store_true", help="Disable entity/attribute typing")
    parser.add_argument("--no-context", action="store_true", help="Disable context retrieval")
    parser.add_argument("--start-at-index", type=int, default=0, help="Index of the first question to evaluate (0-based, inclusive)")
    parser.add_argument("--end-at-index", type=int, default=None, help="Index of the last question to evaluate (0-based, exclusive); defaults to end of list")
    parser.add_argument("--resume-dir", type=Path, default=None, help="Path to an existing output directory to resume. Auto-detects resume index from existing results.")
    parser.add_argument("--rerun-from-question", type=int, default=None, help="When resuming, discard results from this question number onward (1-based, matches [N/total] display) and re-run from there. Requires --resume-dir.")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature for the model (default: 0.0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.resume_dir:
        output_dir = args.resume_dir
        resume = True
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        folder_name = f"evaluation_{Path(args.input_file).stem}_{timestamp}_{args.lang}_{args.model_name.replace('/', '-').replace(':', '-')}_t{args.temp}"
        if args.no_translate:
            folder_name += "_no_translate"
        if args.no_icl:
            folder_name += "_no_icl"
        if args.no_eat:
            folder_name += "_no_eat"
        if args.no_context:
            folder_name += "_no_context"
        if args.test_name:
            folder_name += f"_{args.test_name}"
        output_dir = Path("test_results") / folder_name
        resume = False

    setup_logging(output_dir)
    evaluate(
        input_file=args.input_file,
        output_dir=output_dir,
        api_url=args.api_url,
        sparql_endpoint=args.sparql_endpoint,
        lang=args.lang,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
        num_ports=args.num_ports,
        low_port=args.low_port,
        model_name=args.model_name,
        log_calls=args.log_calls,
        temperature=args.temp,
        use_translate=not args.no_translate,
        use_llm_translate=args.use_llm_translate,
        use_icl=not args.no_icl,
        use_eat=not args.no_eat,
        use_context=not args.no_context,
        cost_prompt=args.cost_prompt,
        cost_completion=args.cost_completion,
        start_at_index=args.start_at_index,
        end_at_index=args.end_at_index,
        resume=resume,
        rerun_from_question=args.rerun_from_question,
    )


if __name__ == "__main__":
    main()