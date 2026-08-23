
import re
from typing import List, Optional, Tuple
import sympy
from sympy.parsing.sympy_parser import parse_expr
from app.config import settings
_VAR_EQUALITY_RE = re.compile(r"\b([a-zA-Z])\s*=\s*(-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+)?)")
_BARE_ANSWER_RE = re.compile(
    r"(?:is|equals|answer(?:\s+is)?|=)\s*:?\s*(-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+)?)",
    re.IGNORECASE,
)

def _parse_canonical(canonical_answer: str) -> List[sympy.core.Expr]:

    values = []
    for part in canonical_answer.split(","):
        part = part.strip()
        if "=" in part:
            part = part.split("=", 1)[1].strip()
        try:
            values.append(parse_expr(part))
        except Exception:
            continue
    return values


def _extract_candidates_from_reply(reply: str) -> List[str]:
    candidates = set()
    for m in _VAR_EQUALITY_RE.finditer(reply):
        candidates.add(m.group(2).replace(" ", ""))
    for m in _BARE_ANSWER_RE.finditer(reply):
        candidates.add(m.group(1).replace(" ", ""))
    return list(candidates)


def check_answer_leak(
    reply: str, canonical_answer: Optional[str]
) -> Tuple[bool, Optional[str]]:

    if not canonical_answer:
        return False, None

    canonical_values = _parse_canonical(canonical_answer)
    if not canonical_values:
        return False, None

    candidates = _extract_candidates_from_reply(reply)
    for cand_str in candidates:
        try:
            cand_expr = parse_expr(cand_str)
        except Exception:
            continue
        for canon in canonical_values:
            try:
                diff = sympy.nsimplify(cand_expr - canon)
                if diff == 0:
                    return True, str(cand_expr)
                if abs(float(cand_expr) - float(canon)) < settings.LEAK_NUMERIC_TOLERANCE:
                    return True, str(cand_expr)
            except Exception:
                continue

    return False, None
