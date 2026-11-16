"""Load authored skills at runtime and select them from question/search state."""
from pathlib import Path
import re

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


def load_skills() -> list[dict[str, str]]:
    loaded = []
    for path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        raw = path.read_text(encoding="utf-8")
        name = re.search(r"(?m)^name:\s*(.+)$", raw)
        description = re.search(r"(?m)^description:\s*(.+)$", raw)
        parts = raw.split("---", 2)
        loaded.append({"name": name.group(1).strip() if name else path.parent.name,
                       "description": description.group(1).strip() if description else "",
                       "instructions": parts[-1].strip()})
    return loaded


def select_skills(question: str, state: dict, skills: list[dict[str, str]]) -> list[dict[str, str]]:
    q = question.casefold().strip()
    selected = []
    # Select identity-resolution guidance from the question's requested answer,
    # not from arbitrary clue text. A raw substring check for "who" also matched
    # embedded clauses such as "people who start businesses" and activated the
    # two-source person-answer gate for unrelated questions.
    asks_identity = bool(
        re.match(r"^(?:who|which person|which individual)\b", q)
        or re.search(r"\b(?:full name|first and last name|person['’]s name|individual['’]s name)\b", q)
        or re.search(
            r"\bname of (?:the |an? )?(?:person|individual|scientist|researcher|"
            r"actor|actress|musician|artist|author|composer|athlete|politician)\b",
            q,
        )
        or re.search(r"\bwhat (?:is|was) (?:the )?(?:name|identity) of\b", q)
    )
    if asks_identity:
        selected.extend(skill for skill in skills if skill["name"] == "multi_hop_identity")
    if state.get("last_search_empty") or state.get("low_evidence"):
        selected.extend(skill for skill in skills if skill["name"] == "search_recovery")
    return selected
