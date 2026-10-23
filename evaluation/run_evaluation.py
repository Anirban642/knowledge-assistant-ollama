"""Runs a fixed set of questions against the running API and reports how many answers are correct.

Usage (the API must be running and the sample handbook indexed):

    export EVAL_API_KEY=<a user or admin API key>
    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --filter notice      # only matching cases

The checks are keyword based: they catch wrong numbers and wrong refusals, but they cannot judge wording.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_QUESTIONS_FILE = Path(__file__).resolve().parent / "questions.json"
DEFAULT_TIMEOUT_SECONDS = 300.0  # the first answer is slow while Ollama loads the model
SESSION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
ANSWER_PREVIEW_CHARS = 300

CATEGORY_IN_SCOPE = "In-scope questions"
CATEGORY_OUT_OF_SCOPE = "Out-of-scope questions (must be refused)"
CATEGORY_CONVERSATION = "Follow-up conversations"


class EvaluationError(Exception):
    """A problem with the evaluation setup itself (not a failed question)."""


@dataclass(frozen=True)
class EvalCase:
    """One question and what a correct response looks like."""

    case_id: str
    question: str
    should_answer: bool = True  # False means the assistant must refuse
    must_contain: list[str] = field(default_factory=list)  # every entry must appear; "a|b" means a OR b
    must_not_contain: list[str] = field(default_factory=list)
    session: str | None = None  # cases with the same session name share one conversation


@dataclass
class CaseResult:
    """What happened when a case was run."""

    case: EvalCase
    answer: str
    answered_from_documents: bool
    standalone_question: str
    seconds: float
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


# ---------- loading and checking (pure functions, covered by unit tests) ----------

def load_cases(path: Path) -> list[EvalCase]:
    """Read and validate the question file."""
    try:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"Cannot read the questions file {path}: {error}") from error

    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for raw in raw_cases:
        try:
            case = EvalCase(
                case_id=raw["id"],
                question=raw["question"],
                should_answer=raw.get("should_answer", True),
                must_contain=raw.get("must_contain", []),
                must_not_contain=raw.get("must_not_contain", []),
                session=raw.get("session"),
            )
        except KeyError as error:
            raise EvaluationError(f"Each case needs an 'id' and a 'question' (problem: {error}).") from error

        if case.case_id in seen_ids:
            raise EvaluationError(f"Duplicate case id '{case.case_id}'.")
        if case.session and not SESSION_NAME_PATTERN.match(case.session):
            raise EvaluationError(f"Invalid session name '{case.session}' in case '{case.case_id}'.")
        seen_ids.add(case.case_id)
        cases.append(case)
    return cases


def select_cases(cases: list[EvalCase], text: str | None) -> list[EvalCase]:
    """Keep only cases whose id or session name contains the given text."""
    if not text:
        return cases
    needle = text.lower()
    return [
        case for case in cases
        if needle in case.case_id.lower() or (case.session and needle in case.session.lower())
    ]


def contains_any_alternative(text: str, option: str) -> bool:
    """True if the text contains any '|' separated alternative of the option, ignoring case."""
    lowered = text.lower()
    return any(alternative.strip().lower() in lowered for alternative in option.split("|"))


def evaluate_answer(case: EvalCase, answer: str, answered_from_documents: bool) -> list[str]:
    """Return the reasons the response is wrong. An empty list means the case passed."""
    failures: list[str] = []

    if answered_from_documents != case.should_answer:
        expected = "an answer" if case.should_answer else "a refusal"
        actual = "an answer" if answered_from_documents else "a refusal"
        failures.append(f"expected {expected} but got {actual}")

    if case.should_answer and answered_from_documents:  # content checks only make sense for real answers
        failures.extend(
            f"answer is missing '{option}'"
            for option in case.must_contain
            if not contains_any_alternative(answer, option)
        )
        failures.extend(
            f"answer must not contain '{option}'"
            for option in case.must_not_contain
            if contains_any_alternative(answer, option)
        )
    return failures


def categorize(case: EvalCase) -> str:
    """Group a case for the summary."""
    if not case.should_answer:
        return CATEGORY_OUT_OF_SCOPE
    return CATEGORY_CONVERSATION if case.session else CATEGORY_IN_SCOPE


def pass_rate(results: list[CaseResult]) -> float:
    """Fraction of cases that passed (0 to 1)."""
    return sum(result.passed for result in results) / len(results)


# ---------- talking to the API ----------

class EvalClient:
    """Small wrapper around the HTTP API so the rest of the script reads like plain English."""

    def __init__(self, base_url: str, api_key: str, timeout_seconds: float) -> None:
        self._http = httpx.Client(base_url=base_url, headers={"X-API-Key": api_key}, timeout=timeout_seconds)

    def close(self) -> None:
        self._http.close()

    def check_ready(self) -> None:
        """Stop early if Ollama, a model or the vector database is not usable."""
        response = self._http.get("/health/ready")
        if response.status_code != 200:
            failing = [check["name"] for check in response.json().get("checks", []) if not check["healthy"]]
            raise EvaluationError(
                f"The API is not ready (unhealthy: {', '.join(failing) or 'unknown'}). "
                "Open /api/v1/health/ready for details."
            )

    def list_documents(self) -> list[str]:
        response = self._http.get("/documents")
        self._raise_if_rejected(response)
        response.raise_for_status()
        return [document["source"] for document in response.json()["documents"]]

    def ask(self, question: str, session_id: str | None) -> httpx.Response:
        payload: dict[str, str] = {"question": question}
        if session_id:
            payload["session_id"] = session_id
        response = self._http.post("/chat", json=payload)
        self._raise_if_rejected(response)
        return response

    def clear_session(self, session_id: str) -> None:
        """Best-effort cleanup. A 404 just means the session was never created."""
        try:
            self._http.delete(f"/chat/sessions/{session_id}")
        except httpx.HTTPError:
            pass

    @staticmethod
    def _raise_if_rejected(response: httpx.Response) -> None:
        if response.status_code in (401, 403):
            raise EvaluationError(
                f"The API rejected the key (HTTP {response.status_code}). Check EVAL_API_KEY or --api-key."
            )


def _error_message(response: httpx.Response) -> str:
    """Pull the message out of the API's standard error body, falling back to the raw text."""
    try:
        return response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return response.text[:200]


# ---------- running and reporting ----------

def run_case(client: EvalClient, case: EvalCase, session_id: str | None) -> CaseResult:
    """Ask one question and grade the response."""
    started_at = time.perf_counter()
    response = client.ask(case.question, session_id)
    seconds = time.perf_counter() - started_at

    if response.status_code != 200:
        return CaseResult(
            case=case,
            answer="",
            answered_from_documents=False,
            standalone_question=case.question,
            seconds=seconds,
            failures=[f"API returned HTTP {response.status_code}: {_error_message(response)}"],
        )

    body = response.json()
    return CaseResult(
        case=case,
        answer=body["answer"],
        answered_from_documents=body["answered_from_documents"],
        standalone_question=body["standalone_question"],
        seconds=seconds,
        failures=evaluate_answer(case, body["answer"], body["answered_from_documents"]),
    )


def run_cases(client: EvalClient, cases: list[EvalCase]) -> list[CaseResult]:
    """Run every case in order. Sessions get unique names per run and are cleaned up at the end."""
    run_id = uuid.uuid4().hex[:6]
    session_ids = {case.session: f"eval-{run_id}-{case.session}" for case in cases if case.session}

    results: list[CaseResult] = []
    try:
        for position, case in enumerate(cases, start=1):
            session_id = session_ids[case.session] if case.session else None
            result = run_case(client, case, session_id)
            results.append(result)
            print_case_line(position, len(cases), result)
    finally:
        for session_id in session_ids.values():
            client.clear_session(session_id)
    return results


def _preview(text: str) -> str:
    return " ".join(text.split())[:ANSWER_PREVIEW_CHARS]


def print_case_line(position: int, total: int, result: CaseResult) -> None:
    """Print one result immediately, so progress is visible during the long run."""
    status = "PASS" if result.passed else "FAIL"
    print(f"[{position:>2}/{total}] {status}  {result.case.case_id}  ({result.seconds:.1f}s)", flush=True)
    if result.passed:
        return
    for failure in result.failures:
        print(f"         - {failure}")
    print(f"         question used for search: {result.standalone_question}")
    print(f"         answer: {_preview(result.answer)}", flush=True)


def print_report(results: list[CaseResult]) -> None:
    """Print the summary: overall score, score per group, median speed and the failed ids."""
    passed = sum(result.passed for result in results)
    print("\n" + "=" * 60)
    print(f"Passed {passed}/{len(results)} ({pass_rate(results):.1%})")

    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        by_category[categorize(result.case)].append(result)
    for category, items in by_category.items():
        print(f"  {category}: {sum(item.passed for item in items)}/{len(items)}")

    # The median ignores the one slow answer while Ollama loads the model.
    print(f"Median response time: {statistics.median(result.seconds for result in results):.1f}s")

    failed = [result.case.case_id for result in results if not result.passed]
    if failed:
        print("Failed cases: " + ", ".join(failed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the knowledge assistant against a fixed question set.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL including /api/v1")
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. Prefer the EVAL_API_KEY environment variable so the key stays out of your shell history.",
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_FILE, help="Path to the questions file")
    parser.add_argument("--filter", default=None, help="Only run cases whose id or session name contains this text")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="Exit with code 1 when the pass rate (0 to 1) is lower than this. Useful in CI.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="Seconds to wait per answer")
    return parser.parse_args()


def main() -> int:
    # Answers can contain characters (such as the rupee sign) that some Windows consoles cannot print.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    api_key = args.api_key or os.environ.get("EVAL_API_KEY")
    if not api_key:
        print("No API key. Set the EVAL_API_KEY environment variable or pass --api-key.")
        return 2

    client = EvalClient(args.base_url, api_key, args.timeout)
    try:
        cases = select_cases(load_cases(args.questions), args.filter)
        if not cases:
            raise EvaluationError("No questions matched the filter.")

        client.check_ready()
        documents = client.list_documents()
        if not documents:
            raise EvaluationError("The knowledge base is empty. Upload samples/company_handbook.txt first.")

        print(f"Indexed documents: {', '.join(documents)}")
        print(f"Running {len(cases)} question(s) against {args.base_url}\n", flush=True)
        results = run_cases(client, cases)
    except httpx.ConnectError:
        print(f"Cannot reach the API at {args.base_url}. Is uvicorn running?")
        return 2
    except httpx.TimeoutException:
        print("The API took too long to answer. Check that Ollama is running, or raise --timeout.")
        return 2
    except EvaluationError as error:
        print(f"Error: {error}")
        return 2
    finally:
        client.close()

    print_report(results)
    return 0 if pass_rate(results) >= args.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())