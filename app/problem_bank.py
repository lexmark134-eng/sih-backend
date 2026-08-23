
from typing import Optional, List, Dict, Any
import sympy
from sympy.parsing.sympy_parser import parse_expr
from app.vectorstore import get_problem_bank_store


def solve_equation_symbolically(equation_str: str, variable: str = "x") -> Optional[List[str]]:
    """
    Attempts to solve a simple equation string like 'x**2 - 9 = 0' or
    '2*x + 4 = 10' using sympy, independent of any LLM call.
    Returns a list of solution strings, or None if it can't be parsed/solved.
    Only handles clean symbolic/algebraic input — for messier word problems,
    supply canonical_answer directly when adding the problem (see add_problem).
    """
    try:
        var = sympy.symbols(variable)
        if "=" in equation_str:
            lhs, rhs = equation_str.split("=", 1)
            expr = parse_expr(lhs) - parse_expr(rhs)
        else:
            expr = parse_expr(equation_str)
        solutions = sympy.solve(expr, var)
        return [str(s) for s in solutions]
    except Exception:
        return None


def add_problem(
    topic: str,
    problem_text: str,
    canonical_answer: str,
    variable: str = "x",
    source: str = "instructor",
) -> Dict[str, Any]:
    """
    canonical_answer is stored as a plain string, e.g. "3" or "x = 3, x = -3"
    or "12.5". Keep it in a form answer_leak.py's parser can extract numbers
    /expressions from (see extract_values_from_text there — same parser is
    reused on both sides for consistency).
    """
    store = get_problem_bank_store()
    store.add_texts(
        texts=[problem_text],
        metadatas=[{
            "topic": topic,
            "canonical_answer": canonical_answer,
            "variable": variable,
            "source": source,
        }],
    )
    return {"problem_text": problem_text, "canonical_answer": canonical_answer}


def add_problem_auto_solve(
    topic: str,
    problem_text: str,
    equation_str: str,
    variable: str = "x",
) -> Optional[Dict[str, Any]]:
    """Convenience: solve the equation with sympy, then store the result."""
    solutions = solve_equation_symbolically(equation_str, variable)
    if not solutions:
        return None
    canonical = ", ".join(f"{variable} = {s}" for s in solutions)
    return add_problem(topic, problem_text, canonical, variable)


def match_problem(student_message: str, topic: str, threshold: float) -> Optional[Dict[str, Any]]:
    """
    Finds the closest stored problem to what the student is working on.
    Returns None if nothing clears the relevance threshold (i.e. this is a
    novel problem not in the bank — the leak check is simply skipped for it,
    since there's no independently-verified answer to check against).
    """
    store = get_problem_bank_store()
    results = store.similarity_search_with_relevance_scores(
        student_message, k=1, filter={"topic": topic}
    )
    if not results:
        return None
    doc, score = results[0]
    if score < threshold:
        return None
    return {
        "problem_text": doc.page_content,
        "canonical_answer": doc.metadata.get("canonical_answer"),
        "variable": doc.metadata.get("variable", "x"),
        "score": score,
    }
