"""Bounded iterative agent with explicit state, runtime skills, and trajectory logging."""
from __future__ import annotations

import json
import os
import re

from openai import OpenAI

from .config import settings
from .skills import load_skills, select_skills

ACTION_TOOLS = [
    {"type": "function", "function": {
        "name": "search", "description": (f"Search the fixed corpus with one primary query and up to "
                                                f"{settings.max_additional_queries} independent clue queries."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "additional_queries": {"type": "array", "items": {"type": "string"},
                                   "maxItems": settings.max_additional_queries}},
                       "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "open", "description": "Open a document returned by search.",
        "parameters": {"type": "object", "properties": {"doc_id": {"type": "string"}},
                       "required": ["doc_id"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "find", "description": "Find an exact phrase in a retrieved document.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"}, "phrase": {"type": "string"}},
            "required": ["doc_id", "phrase"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "answer", "description": "Return the final concise answer and cite opened source documents that support it. Use an empty list only when abstaining.",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string"},
            "supporting_doc_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
            "evidence_ledger": {"type": "array", "items": {"type": "object", "properties": {
                "clue": {"type": "string"}, "doc_id": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["clue", "doc_id", "quote"], "additionalProperties": False}, "maxItems": 6}},
                       "required": ["answer", "supporting_doc_ids"], "additionalProperties": False}}},
]

if settings.structured_evidence_ledger:
    ACTION_TOOLS[-1]["function"]["parameters"]["required"].append("evidence_ledger")

IDENTITY_CITATION_RULE = (
    f"For a multi-hop person identification, cite at least {settings.min_identity_sources} distinct opened "
    f"document{'s' if settings.min_identity_sources != 1 else ''} supporting independently verified clues. "
    "A single source may support multiple clues only when its text explicitly states each clue. "
    "If that evidence requirement cannot be met, return a concise insufficient-evidence answer with an empty citation list."
)

SYSTEM_PROMPT = f"""You answer research questions using only the supplied fixed-corpus tools and evidence.
For every turn, call exactly one of the provided functions: search, open, find, or answer.
Do not write a free-form message or call multiple functions in one turn.
Use up to three search actions. Each search action may include up to {settings.max_additional_queries} additional_queries, each
targeting a different clue path; the retrieval tool will merge their results. For multi-hop questions,
make the primary query target the most discriminative clue and use additional queries for independent
clues, rather than submitting near-duplicate phrasings. After each action, inspect the returned
documents and open the strongest candidates. A new search action should address a missing clue.
Build queries only from details in the question or facts verified in retrieved evidence. Do not add
specific dates, names, countries, awards, or events that have not been stated or verified. Use genuine
synonyms, but do not conflate related scientific terms: cytokinin and cytokine are different terms,
and interferon is a cytokine rather than a synonym for cytokinin. Treat all query expansions as search
hypotheses, not established facts. You may use all three search actions before opening a document when
each targets a distinct unresolved clue. Open promising candidates as they appear, and stop searching
when the search budget is exhausted.
Before answering, keep an evidence ledger: which clues are supported, which remain unresolved, and
which source supports each candidate. Search-hit provenance indicates which query retrieved a document;
it is a ranking hint, not proof that the document supports that clue. Verify the claim in the document text.
When a candidate appears, search for that candidate together with a separate distinctive clue to verify
the identity. {IDENTITY_CITATION_RULE} Cite only documents you opened.
Do not answer "insufficient evidence" while
searches remain and a specific unresolved clue can still be searched. After the search budget is
exhausted, inspect the strongest current candidates and then answer from evidence. Do not invent
facts. Keep the final answer concise. For questions asking for a person,
organization, place, or other entity name, return only the requested name without explanations,
qualifiers, or extra facts; for other questions, return only the shortest answer that fully answers it."""

# Experimental switch used by the optimization notebook. The default preserves
# the retained agent behavior; pilots can enable this in a subprocess without
# changing the project's .env file.
STRICT_FINAL_ANSWER_FORMAT = os.getenv("STRICT_FINAL_ANSWER_FORMAT", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
if STRICT_FINAL_ANSWER_FORMAT:
    SYSTEM_PROMPT += """
For the final answer, reproduce the shortest canonical answer supported by the cited evidence.
Do not append explanations, parenthetical aliases, confidence language, or extra descriptive clauses.
For dates, measurements, percentages, and names, preserve the source's written form and precision;
do not convert units, date formats, or shorten full names. Return only the requested value or name."""

if settings.explicit_candidate_state:
    SYSTEM_PROMPT += """
Candidate state is authoritative. Never call open for a document ID already listed in opened_docs;
that document's text is already available in evidence_excerpts. To inspect another source, select an
ID from unopened_candidates. If no unopened candidates remain, use find on an opened source, search
for a new unresolved clue when budget remains, or answer from the evidence."""

if settings.structured_evidence_ledger:
    SYSTEM_PROMPT += """
Before answering, build a compact evidence_ledger in the answer function arguments. Each entry must
contain one distinct question clue, the ID of an opened document, and a short verbatim quote from that
document that supports the clue. Quotes must be copied exactly from opened text. For multi-hop person
identification, provide at least two distinct clue entries; one opened document may support both only
if the quoted text explicitly supports both. Include every ledger document in supporting_doc_ids.
Never use search-result snippets as proof. If the required clues cannot be verified, search an unresolved
clue while budget remains; otherwise abstain with empty citations and an empty ledger. Do not expose
private reasoning; return only the concise answer and structured evidence fields."""

ACTION_NAMES = {"search", "open", "find", "answer"}


def validate_action(value: object) -> dict:
    """Validate one decoded function call before executing it."""
    if not isinstance(value, dict):
        return {"action": "invalid", "error": "Function arguments must be a JSON object."}
    kind = value.get("action")
    if kind not in ACTION_NAMES:
        return {"action": "invalid", "error": "The action must be search, open, find, or answer."}

    required = {
        "search": ("query",),
        "open": ("doc_id",),
        "find": ("doc_id", "phrase"),
        "answer": ("answer",),
    }[kind]
    normalized = {"action": kind}
    for field in required:
        item = value.get(field)
        if not isinstance(item, str) or not item.strip():
            return {"action": "invalid", "error": f"Action '{kind}' requires a non-empty string field '{field}'."}
        normalized[field] = item.strip()

    if kind == "search" and "additional_queries" in value:
        additions = value["additional_queries"]
        if not isinstance(additions, list) or len(additions) > settings.max_additional_queries:
            return {"action": "invalid", "error": (
                f"'additional_queries' must be a list of at most {settings.max_additional_queries} strings.")}
        cleaned = []
        for item in additions:
            if not isinstance(item, str) or not item.strip():
                return {"action": "invalid", "error": "Every additional query must be a non-empty string."}
            cleaned.append(item.strip())
        normalized["additional_queries"] = cleaned

    if kind == "answer":
        sources = value.get("supporting_doc_ids")
        if not isinstance(sources, list) or len(sources) > 4:
            return {"action": "invalid", "error": "'supporting_doc_ids' must be a list of at most four document IDs."}
        if any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in sources):
            return {"action": "invalid", "error": "Every supporting document ID must be a non-empty string."}
        normalized["supporting_doc_ids"] = list(dict.fromkeys(doc_id.strip() for doc_id in sources))
        if "evidence_ledger" in value:
            ledger = value["evidence_ledger"]
            if not isinstance(ledger, list) or len(ledger) > 6:
                return {"action": "invalid", "error": "'evidence_ledger' must be a list of at most six entries."}
            cleaned_ledger = []
            for entry in ledger:
                if not isinstance(entry, dict):
                    return {"action": "invalid", "error": "Every evidence_ledger entry must be an object."}
                clue, doc_id, quote = entry.get("clue"), entry.get("doc_id"), entry.get("quote")
                if not all(isinstance(item, str) and item.strip() for item in (clue, doc_id, quote)):
                    return {"action": "invalid", "error": "Each evidence entry needs non-empty clue, doc_id, and quote strings."}
                if len(clue) > 500 or len(doc_id) > 200 or len(quote) > 800:
                    return {"action": "invalid", "error": "Evidence clue, doc_id, or quote exceeds its length limit."}
                cleaned_ledger.append({"clue": clue.strip(), "doc_id": doc_id.strip(), "quote": quote.strip()})
            normalized["evidence_ledger"] = cleaned_ledger
        elif settings.structured_evidence_ledger:
            return {"action": "invalid", "error": "'evidence_ledger' is required for a supported answer."}

    max_lengths = {"query": 500, "doc_id": 200, "phrase": 500, "answer": 4000}
    for field, limit in max_lengths.items():
        if field in normalized and len(normalized[field]) > limit:
            return {"action": "invalid", "error": f"Field '{field}' exceeds the {limit}-character limit."}
    if kind == "search" and any(len(query) > 500 for query in normalized.get("additional_queries", [])):
        return {"action": "invalid", "error": "An additional query exceeds the 500-character limit."}
    return normalized


def parse_tool_action(tool_calls) -> dict:
    """Convert the first provider function call into a validated internal action."""
    if not tool_calls:
        return {"action": "invalid", "error": "The model did not return a function call."}
    # DeepSeek's `tool_choice="required"` may return more than one call. Execute
    # only the first action in this bounded turn; the remaining choices are ignored.
    call = tool_calls[0]
    if not getattr(call, "id", None):
        return {"action": "invalid", "error": "The function call is missing its provider call ID."}
    name = call.function.name
    if name not in ACTION_NAMES:
        return {"action": "invalid", "error": "The function must be search, open, find, or answer."}
    try:
        arguments = json.loads(call.function.arguments or "")
    except (json.JSONDecodeError, TypeError):
        return {"action": "invalid", "error": "Function arguments were not valid JSON."}
    if not isinstance(arguments, dict):
        return {"action": "invalid", "error": "Function arguments must be a JSON object."}
    return validate_action({"action": name, **arguments})


class SearchAgent:
    def __init__(self, tools, api_key: str | None = None, progress: bool = False):
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is missing. Set it in the project .env file.")
        self.client = OpenAI(api_key=key, base_url=settings.base_url,
                             timeout=180.0, max_retries=0)
        self.tools = tools
        self.skills = load_skills()
        self.progress = progress

    @staticmethod
    def _allowed_years(text: str) -> set[int]:
        """Expand explicit years, ranges, and decade descriptions into allowed years."""
        allowed = {int(year) for year in re.findall(r"\b(?:18|19|20)\d{2}\b", text)}
        lowered = text.casefold()
        for match in re.finditer(r"\bbetween\s+((?:18|19|20)\d{2})\s+and\s+((?:18|19|20)\d{2})\b", lowered):
            start, end = int(match.group(1)), int(match.group(2))
            range_tail = lowered[match.end():match.end() + 100]
            if "exclusive" in range_tail:
                start += 1
                end -= 1
            if start <= end and end - start <= 100:
                allowed.update(range(start, end + 1))

        decade_pattern = r"\b((?:18|19|20)\d)0s\b"
        for match in re.finditer(decade_pattern, lowered):
            decade_start = int(match.group(1) + "0")
            prefix = lowered[max(0, match.start() - 48):match.start()]
            if re.search(r"first decade of (?:the )?$", prefix):
                begin, finish = decade_start, decade_start + 9
            elif re.search(r"early\s+$", prefix):
                begin, finish = decade_start, decade_start + 4
            elif re.search(r"mid(?:dle)?\s+(?:of\s+)?$", prefix):
                begin, finish = decade_start + 3, decade_start + 7
            elif re.search(r"late\s+$", prefix):
                begin, finish = decade_start + 5, decade_start + 9
            else:
                begin, finish = decade_start, decade_start + 9
            allowed.update(range(begin, finish + 1))
        return allowed

    @classmethod
    def _sanitize_query_years(cls, query: str, source_text: str,
                              explicit_year_text: str | None = None) -> tuple[str, list[dict[str, int]]]:
        """Keep query dates at the precision supported by the question/evidence."""
        allowed = cls._allowed_years(source_text)
        # Exact years from an opened document may be unrelated to the clue that
        # generated this query. Preserve exact precision only when the question
        # itself states it; evidence can still broaden the allowed decade range.
        explicit_text = source_text if explicit_year_text is None else explicit_year_text
        explicit = {int(year) for year in re.findall(r"\b(?:18|19|20)\d{2}\b", explicit_text)}
        changes: list[dict[str, int]] = []

        def replace(match: re.Match) -> str:
            year = int(match.group(0))
            if year in explicit:
                return match.group(0)
            if year not in allowed:
                changes.append({"year": year, "decade": 0})
                return ""
            decade = (year // 10) * 10
            changes.append({"year": year, "decade": decade})
            # Keep decade qualifiers when the source clue gives them (e.g. early 1990s).
            decade_pattern = rf"\b(early|mid(?:dle)?|late)\s+(?:the\s+)?{decade}s\b"
            qualifier = re.search(decade_pattern, source_text, re.IGNORECASE)
            if qualifier:
                return f"{qualifier.group(1)} {decade}s"
            return f"{decade}s"

        sanitized = re.sub(r"\b(?:18|19|20)\d{2}\b", replace, query)
        sanitized = re.sub(r"\s+", " ", sanitized).strip(" ,;:-")
        return sanitized, changes

    def _state_message(self, state: dict, final_turn: bool, next_turn_final: bool = False) -> dict:
        opened = set(state["opened_docs"])
        opened_evidence = [item for item in state["evidence"] if item["doc_id"] in opened]
        useful_evidence = opened_evidence[-4:] if opened_evidence else state["evidence"][-6:]
        state_payload = {
            "visited_docs": state["visited_docs"][-20:], "queries": state["queries"][-8:],
            "blocked_queries": state["blocked_queries"][-6:],
            "opened_docs": state["opened_docs"][-8:], "search_calls": state["search_calls"],
            "finds": state["finds"][-8:],
            "evidence_doc_ids": list(dict.fromkeys(x["doc_id"] for x in state["evidence"])),
            "evidence_excerpts": [{"doc_id": x["doc_id"], "matched_queries": x.get("matched_queries", []),
                                  "query_ranks": x.get("query_ranks", []), "text": x["text"][:800]}
                                  for x in useful_evidence],
            "last_search_empty": state["last_search_empty"], "low_evidence": state["low_evidence"]
        }
        if settings.explicit_candidate_state:
            candidate_map: dict[str, dict] = {}
            for item in state["evidence"]:
                doc_id = str(item["doc_id"])
                if doc_id in opened or item.get("score") is None:
                    continue
                current = candidate_map.get(doc_id)
                if current is None:
                    candidate_map[doc_id] = {
                        "doc_id": doc_id, "score": item["score"],
                        "matched_queries": list(dict.fromkeys(item.get("matched_queries", []))),
                        "query_ranks": item.get("query_ranks", []),
                    }
                else:
                    current["score"] = max(current["score"], item["score"])
                    current["matched_queries"] = list(dict.fromkeys(
                        current["matched_queries"] + item["matched_queries"]))
                    current["query_ranks"] = current["query_ranks"] + [
                        rank for rank in item.get("query_ranks", [])
                        if rank not in current["query_ranks"]]
            state_payload["unopened_candidates"] = sorted(
                candidate_map.values(), key=lambda item: (-item["score"], item["doc_id"]))[:12]
        content = "Current state: " + json.dumps(state_payload, ensure_ascii=False)
        if settings.explicit_candidate_state:
            content += ("\nCandidate handling: opened_docs are already inspected; never request them again. "
                        "Choose an ID from unopened_candidates to inspect a new document.")
        if state["search_calls"] >= settings.max_search_calls:
            content += "\nSearch budget exhausted: do not search again; open/find in current candidate documents or answer."
        if final_turn:
            content += ("\nFINAL TURN: return an answer action now. Do not call search, open, or find. "
                        "If the evidence is insufficient, say so concisely in the answer.")
        elif next_turn_final:
            content += ("\nNEXT TURN IS FINAL: use this turn for the last useful search or document inspection. "
                        "The next turn will require an answer action.")
        return {"role": "system", "content": content}

    @staticmethod
    def _message_size(message: dict) -> int:
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        return len(content) if isinstance(content, str) else len(json.dumps(tool_calls or [], ensure_ascii=False))

    @staticmethod
    def _state_error(action: dict, state: dict, final_turn: bool,
                     minimum_identity_sources: int = 0) -> str | None:
        kind = action["action"]
        if final_turn and kind != "answer":
            return "This is the final turn. Return an answer action; tool actions are not allowed."
        if kind == "search":
            queries = [action["query"], *action.get("additional_queries", [])]
            normalized_queries = [re.sub(r"\s+", " ", q.casefold()) for q in queries]
            if len(set(normalized_queries)) != len(normalized_queries):
                return "Search queries in one action must target distinct clues; remove duplicates."
            prior = {re.sub(r"\s+", " ", value.casefold())
                     for value in [*state["queries"], *state["blocked_queries"]]}
            if state["search_calls"] >= settings.max_search_calls:
                return "Search budget exhausted; inspect candidate documents or answer."
            if any(query in prior for query in normalized_queries):
                return "A query was already used; replace it with a new, unresolved clue query."
        if kind == "answer":
            abstention_markers = (
                "insufficient evidence", "cannot verify", "could not verify", "unable to identify",
                "evidence is insufficient", "not enough information", "cannot determine",
            )
            is_abstention = any(marker in action["answer"].casefold() for marker in abstention_markers)
            if is_abstention and not final_turn:
                if state["search_calls"] < settings.max_search_calls:
                    return ("Do not stop early with an insufficient-evidence answer. Use the remaining search budget "
                            "for one specific unresolved clue, or open a promising retrieved document.")
            if not is_abstention:
                cited = set(action.get("supporting_doc_ids", []))
                opened = set(state["opened_docs"])
                if not cited:
                    return "A supported answer must cite at least one opened source document."
                if cited - opened:
                    return "Citations must refer only to documents opened during this run."
                if minimum_identity_sources and len(cited) < minimum_identity_sources:
                    return (f"A multi-hop person answer needs {minimum_identity_sources} distinct opened "
                            "documents supporting separate clues. "
                            "Open another candidate source if a turn remains, or abstain if the evidence is insufficient.")
                if settings.structured_evidence_ledger:
                    ledger = action.get("evidence_ledger", [])
                    if not ledger:
                        return "A supported answer needs an evidence_ledger with clue, opened doc_id, and verbatim quote entries."
                    if any(entry["doc_id"] not in opened for entry in ledger):
                        return "Every evidence_ledger doc_id must refer to a document opened during this run."
                    if not {entry["doc_id"] for entry in ledger}.issubset(cited):
                        return "Cite every document used by the evidence_ledger in supporting_doc_ids."
                    opened_text = {}
                    for item in state["evidence"]:
                        if item["doc_id"] in opened:
                            opened_text[item["doc_id"]] = re.sub(r"\s+", " ", item["text"]).casefold()
                    distinct_clues = set()
                    for entry in ledger:
                        clue_key = re.sub(r"\s+", " ", entry["clue"]).casefold()
                        quote_key = re.sub(r"\s+", " ", entry["quote"]).casefold()
                        distinct_clues.add(clue_key)
                        if quote_key not in opened_text.get(entry["doc_id"], ""):
                            return (f"The quote for clue '{entry['clue']}' was not found verbatim in opened document "
                                    f"{entry['doc_id']}. Replace it with an exact excerpt from that document.")
                    if minimum_identity_sources and len(distinct_clues) < 2:
                        return "A multi-hop identity needs at least two distinct question clues in evidence_ledger, each with an opened-source quote."
        if kind in {"open", "find"}:
            if action["doc_id"] not in state["visited_docs"]:
                return "That document ID is not among the retrieved candidates. Use an ID from search results."
            if kind == "open" and action["doc_id"] in state["opened_docs"]:
                if settings.explicit_candidate_state:
                    unopened = [str(item["doc_id"]) for item in state["evidence"]
                                if item.get("score") is not None and str(item["doc_id"]) not in state["opened_docs"]]
                    suggestions = ", ".join(dict.fromkeys(unopened[:5])) or "none"
                    return (f"Document {action['doc_id']} is already open; do not reopen it. Its text is in the "
                            f"evidence ledger. Unopened candidate IDs include: {suggestions}. Use one of them, "
                            "use find on an opened source, or answer.")
                return "That document is already open. Inspect its evidence, open another candidate, or answer."
            if kind == "find" and (action["doc_id"], action["phrase"].casefold()) in state["finds"]:
                return "That exact phrase was already checked in this document. Try a different phrase or answer."
        return None

    def _fallback_open(self, state: dict, trajectory: list[dict], step: int,
                       origin_action: str, reason: str) -> dict:
        """Open the best retrieved candidate after a state-invalid tool action."""
        opened = set(state["opened_docs"])
        candidates = [item for item in state["evidence"]
                      if item.get("score") is not None and item["doc_id"] not in opened]
        if not candidates:
            return {"error": reason +
                    " No unopened retrieved candidates remain; inspect current evidence or answer."}

        candidate = max(candidates, key=lambda item: item.get("score", 0.0))
        observation = self.tools.open_document(candidate["doc_id"])
        if "error" in observation:
            return observation

        doc_id = observation["id"]
        observation["notice"] = (
            f"The requested action could not run ({reason}); the agent opened the highest-scoring "
            "retrieved document that had not been inspected. Do not repeat the same open request; "
            "use the opened document's evidence or select another unopened candidate."
            if settings.explicit_candidate_state else
            "The requested action could not run; the agent opened the highest-scoring "
            "retrieved document that had not been inspected."
        )
        if doc_id not in state["visited_docs"]:
            state["visited_docs"].append(doc_id)
        if doc_id not in state["opened_docs"]:
            state["opened_docs"].append(doc_id)
        source = next((item for item in reversed(state["evidence"])
                       if item["doc_id"] == doc_id and item.get("score") is not None), {})
        state["evidence"].append({
            "doc_id": doc_id, "text": observation["text"],
            "url": observation.get("url", ""), "score": source.get("score"),
            "matched_queries": source.get("matched_queries", []),
            "query_ranks": source.get("query_ranks", []),
        })
        state["low_evidence"] = False
        trajectory.append({"step": step, "event": "invalid_action_fallback_open",
                           "origin_action": origin_action, "reason": reason,
                           "doc_id": doc_id, "score": candidate.get("score")})
        return observation

    def run(self, question: str) -> dict:
        state = {"visited_docs": [], "opened_docs": [], "evidence": [], "queries": [], "finds": [],
                 "blocked_queries": [], "search_calls": 0,
                 "last_search_empty": False, "low_evidence": False}
        selected_skills: list[str] = []
        selected_skill_defs: dict[str, dict[str, str]] = {}
        trajectory: list[dict] = []
        tool_calls = total_tokens = llm_calls = invalid_action_retries = 0
        history = [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": question}]
        final_answer = None
        stop_reason = ""

        for step in range(1, settings.max_agent_steps + 1):
            final_turn = step == settings.max_agent_steps
            if self.progress:
                suffix = " (answer required)" if final_turn else ""
                print(f"  Agent step {step}/{settings.max_agent_steps}{suffix}: waiting for DeepSeek...", flush=True)
            for skill in (select_skills(question, state, self.skills) if settings.enable_skills else []):
                if skill["name"] not in selected_skill_defs:
                    selected_skill_defs[skill["name"]] = skill
                    selected_skills.append(skill["name"])
            skill_messages = [{"role": "system", "content": f"Runtime skill {name}:\n{skill['instructions']}"}
                              for name, skill in selected_skill_defs.items()]
            # History after the initial system/user pair consists of complete
            # assistant tool-call + tool-result pairs. Keep the newest two pairs.
            recent = history[2:][-4:]
            state_message = self._state_message(
                state, final_turn, next_turn_final=(step == settings.max_agent_steps - 1))
            messages = [history[0], history[1], *skill_messages, state_message, *recent]
            while sum(self._message_size(item) for item in messages) > settings.max_context_chars and recent:
                # Never split a function call from its tool response. The API
                # rejects a tool result whose preceding assistant tool_calls
                # message was trimmed away.
                del recent[:min(2, len(recent))]
                messages = [history[0], history[1], *skill_messages, state_message, *recent]

            retry_messages: list[dict] = []
            action = None
            accepted_tool_call = None
            semantic_error = None
            raw = ""
            finish_reason = None
            attempt = 0
            request_max_tokens = settings.max_completion_tokens
            while attempt <= settings.max_action_retries:
                response = self.client.chat.completions.create(
                    model=settings.model,
                    messages=[*messages, *retry_messages],
                    tools=ACTION_TOOLS,
                    tool_choice="required",
                    max_tokens=request_max_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                llm_calls += 1
                if response.usage:
                    total_tokens += int(response.usage.total_tokens)
                if not response.choices:
                    response_tool_calls, raw, finish_reason = [], "", None
                else:
                    choice = response.choices[0]
                    response_tool_calls = choice.message.tool_calls or []
                    raw = choice.message.content or ""
                    finish_reason = choice.finish_reason
                candidate = parse_tool_action(response_tool_calls)
                if finish_reason == "length":
                    error = ("The response was truncated at the output token limit. Return one minimal JSON action "
                             "with no extra text or reasoning in the visible response.")
                else:
                    semantic_error = None
                    if candidate.get("action") == "invalid":
                        error = candidate.get("error")
                    else:
                        semantic_error = self._state_error(
                            candidate, state, final_turn,
                            minimum_identity_sources=(settings.min_identity_sources
                                                      if "multi_hop_identity" in selected_skill_defs else 0))
                        # Final-turn tool calls must be corrected. Other well-formed but
                        # state-invalid actions receive an error observation and use one
                        # bounded agent step instead of aborting the entire question.
                        error = semantic_error if final_turn else None
                if error is None:
                    action = candidate
                    if response_tool_calls:
                        accepted_tool_call = response_tool_calls[0]
                    if len(response_tool_calls) > 1:
                        trajectory.append({"step": step, "event": "extra_function_calls_ignored",
                                           "count": len(response_tool_calls) - 1})
                    break

                trajectory.append({"step": step, "event": "invalid_action", "attempt": attempt + 1,
                                   "retrying": attempt < settings.max_action_retries,
                                   "error": error, "finish_reason": finish_reason,
                                   "max_tokens": request_max_tokens})
                if attempt >= settings.max_action_retries:
                    if final_turn:
                        action = {"action": "answer",
                                  "answer": "Insufficient evidence to provide a supported answer.",
                                  "supporting_doc_ids": []}
                        semantic_error = None
                        accepted_tool_call = None
                        trajectory.append({"step": step, "event": "retry_limit_safe_abstention",
                                           "reason": error})
                    else:
                        stop_reason = "invalid_action_retry_limit"
                    break
                invalid_action_retries += 1
                if self.progress:
                    print(f"  Invalid action; retry {attempt + 1}/{settings.max_action_retries}: {error}", flush=True)
                # Keep malformed content out of the conversation. This avoids feeding empty,
                # partial, or unexpectedly long model output back into the next request.
                retry_messages = [{"role": "user", "content": (
                    f"Your previous response could not be used: {error} "
                    "Call exactly one of the provided functions with valid arguments. Do not send a free-form message "
                    "or multiple function calls."
                )}]
                if finish_reason == "length":
                    request_max_tokens = min(request_max_tokens * 2, settings.max_completion_tokens * 4)
                attempt += 1

            if action is None:
                break

            kind = action["action"]
            if self.progress:
                print(f"  DeepSeek returned action: {kind}", flush=True)
            observation = None
            if kind == "search":
                queries = [action["query"], *action.get("additional_queries", [])]
                if semantic_error:
                    if not final_turn:
                        observation = self._fallback_open(
                            state, trajectory, step, "search", semantic_error)
                        if "error" not in observation:
                            tool_calls += 1
                    else:
                        observation = {"error": semantic_error}
                else:
                    opened = set(state["opened_docs"])
                    verified_text = question + " " + " ".join(
                        item["text"] for item in state["evidence"] if item["doc_id"] in opened)
                    allowed_queries, dropped_queries, generalized_queries = [], [], []
                    for query in queries:
                        sanitized_query, changes = self._sanitize_query_years(
                            query, verified_text, explicit_year_text=question)
                        if changes:
                            generalized_queries.append({"original": query, "query": sanitized_query,
                                                         "years": changes})
                        if not sanitized_query:
                            dropped_queries.append(query)
                        else:
                            allowed_queries.append(sanitized_query)
                    if generalized_queries:
                        trajectory.append({"step": step, "event": "query_years_sanitized",
                                           "queries": generalized_queries})
                        state["blocked_queries"].extend(item["original"] for item in generalized_queries)
                    if dropped_queries:
                        state["blocked_queries"].extend(dropped_queries)
                        trajectory.append({"step": step, "event": "unverified_year_queries_dropped",
                                           "queries": dropped_queries})
                    if not allowed_queries:
                        observation = {"error": (
                            "Search skipped because every query added a specific year absent from the question "
                            "and opened evidence. Reformulate without unsupported years.")}
                    elif len(allowed_queries) > 1 and hasattr(self.tools, "search_many"):
                        observation = self.tools.search_many(allowed_queries, exclude_ids=state["visited_docs"])
                        tool_calls += 1
                        state["search_calls"] += 1
                        state["queries"].extend(allowed_queries)
                    else:
                        if allowed_queries:
                            observation = self.tools.search(allowed_queries[0], exclude_ids=state["visited_docs"])
                            tool_calls += 1
                            state["search_calls"] += 1
                            state["queries"].extend(allowed_queries)
                    if allowed_queries:
                        state["last_search_empty"] = len(observation) == 0
                        for item in observation:
                            if item["id"] not in state["visited_docs"]:
                                state["visited_docs"].append(item["id"])
                            state["evidence"].append({"doc_id": item["id"], "text": item["snippet"],
                                                      "url": item["url"], "score": item.get("score"),
                                                      "matched_queries": item.get("matched_queries", queries),
                                                      "query_ranks": item.get("query_ranks", [])})
                        state["low_evidence"] = not state["opened_docs"] and state["search_calls"] >= 2
                        if settings.auto_open_first_candidate and not state["opened_docs"] and observation:
                            for candidate_rank, candidate in enumerate(
                                    observation[:settings.auto_open_candidate_count], start=1):
                                opened_result = self.tools.open_document(candidate["id"])
                                if "error" in opened_result:
                                    continue
                                doc_id = opened_result["id"]
                                state["opened_docs"].append(doc_id)
                                if doc_id not in state["visited_docs"]:
                                    state["visited_docs"].append(doc_id)
                                state["evidence"].append({
                                    "doc_id": doc_id, "text": opened_result["text"],
                                    "url": opened_result.get("url", ""),
                                    "score": candidate.get("score"),
                                    "matched_queries": candidate.get("matched_queries", queries),
                                    "query_ranks": candidate.get("query_ranks", []),
                                })
                                event = ("auto_open_first_candidate" if candidate_rank == 1
                                         else "auto_open_additional_candidate")
                                trajectory.append({"step": step, "event": event, "doc_id": doc_id,
                                                   "rank": candidate_rank, "score": candidate.get("score")})
                                tool_calls += 1
                                state["low_evidence"] = False
            elif kind == "open":
                if semantic_error:
                    if not final_turn and "already open" in semantic_error.casefold():
                        observation = self._fallback_open(
                            state, trajectory, step, "open", semantic_error)
                        if "error" not in observation:
                            tool_calls += 1
                    else:
                        observation = {"error": semantic_error}
                else:
                    observation = self.tools.open_document(action["doc_id"])
                    tool_calls += 1
                    if "error" not in observation:
                        doc_id = observation["id"]
                        if doc_id not in state["visited_docs"]:
                            state["visited_docs"].append(doc_id)
                        if doc_id not in state["opened_docs"]:
                            state["opened_docs"].append(doc_id)
                        source = next((item for item in reversed(state["evidence"])
                                       if item["doc_id"] == doc_id), {})
                        state["evidence"].append({"doc_id": doc_id, "text": observation["text"],
                                                  "url": observation.get("url", ""),
                                                  "score": source.get("score"),
                                                  "matched_queries": source.get("matched_queries", []),
                                                  "query_ranks": source.get("query_ranks", [])})
                        state["low_evidence"] = False
            elif kind == "find":
                if semantic_error:
                    observation = {"error": semantic_error}
                else:
                    observation = self.tools.find_text(action["doc_id"], action["phrase"])
                    tool_calls += 1
                    state["finds"].append((action["doc_id"], action["phrase"].casefold()))
            elif kind == "answer":
                if semantic_error:
                    observation = {"error": semantic_error}
                else:
                    final_answer = action["answer"]
                    trajectory.append({"step": step, "action": action})
                    stop_reason = "answered"
                    break

            trajectory.append({"step": step, "action": action, "observation": observation})
            compact = json.dumps(observation, ensure_ascii=False)
            if len(compact) > 8500:
                compact = compact[:8500] + "...[truncated]"
            if accepted_tool_call is None:
                # Every executable action originates from a validated provider tool call.
                # Keep a defensive fallback for unusual SDK response objects.
                history.extend([{"role": "assistant", "content": json.dumps(action, ensure_ascii=False)},
                                {"role": "user", "content": "Tool result JSON: " + compact}])
            else:
                history.extend([
                    {"role": "assistant", "content": None, "tool_calls": [{
                        "id": accepted_tool_call.id,
                        "type": "function",
                        "function": {"name": accepted_tool_call.function.name,
                                     "arguments": accepted_tool_call.function.arguments},
                    }]},
                    {"role": "tool", "tool_call_id": accepted_tool_call.id, "content": compact},
                ])
            unique = {}
            for item in state["evidence"]:
                key = (item["doc_id"], re.sub(r"\s+", " ", item["text"])[:300])
                unique[key] = item
            unique_items = list(unique.values())
            opened_ids = set(state["opened_docs"])
            opened_items = [item for item in unique_items if item["doc_id"] in opened_ids][-4:]
            opened_ids_kept = {item["doc_id"] for item in opened_items}
            retrieved_items = [item for item in unique_items
                               if item["doc_id"] not in opened_ids_kept][-8:]
            # Preserve opened source evidence across later searches; otherwise recent
            # snippets can evict the documents needed for multi-hop verification.
            state["evidence"] = opened_items + retrieved_items

        if final_answer is None:
            final_answer = "I could not verify a complete answer from the retrieved evidence."
            if not stop_reason:
                stop_reason = "maximum_agent_steps_reached"
            trajectory.append({"step": settings.max_agent_steps, "action": "stop", "reason": stop_reason})
        return {"answer": final_answer, "trajectory": trajectory, "tool_calls": tool_calls,
                "total_tokens": total_tokens, "llm_calls": llm_calls,
                "invalid_action_retries": invalid_action_retries, "stop_reason": stop_reason,
                "state": state, "selected_skills": selected_skills}
