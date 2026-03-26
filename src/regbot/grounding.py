from __future__ import annotations

from typing import Any, Dict, List, Set


def allowed_chunk_ids(chunks: List[Dict[str, Any]]) -> Set[str]:
    return {str(c["id"]) for c in chunks if c.get("id")}


def audit_citation_grounding(
    report: Dict[str, Any],
    allowed_ids: Set[str],
    *,
    strict_min_citations_vs_recommendations: bool = True,
) -> Dict[str, Any]:
    """
    Programmatic grounding check: every citation must reference a chunk_id that was
    actually provided to the model (the retrieved evidence set).
    Optionally require at least as many citations as recommendations (cheap strictness).
    """
    citations = report.get("citations") or []
    recommendations = report.get("recommendations") or []

    invalid_ids: List[str] = []
    valid_ids: List[str] = []
    for c in citations:
        if not isinstance(c, dict):
            invalid_ids.append("<non-object>")
            continue
        cid = c.get("chunk_id")
        if cid is None or str(cid) not in allowed_ids:
            invalid_ids.append(str(cid))
        else:
            valid_ids.append(str(cid))

    issues: List[str] = []
    if recommendations and not citations:
        issues.append("Recommendations are non-empty but citations are empty.")
    if invalid_ids:
        issues.append(f"Invalid chunk_id values in citations: {invalid_ids}")
    if strict_min_citations_vs_recommendations and recommendations:
        if len(citations) < len(recommendations):
            issues.append(
                "Strict mode: need at least as many citations as recommendations "
                f"({len(citations)} citations vs {len(recommendations)} recommendations)."
            )

    ok = len(issues) == 0
    return {
        "ok": ok,
        "issues": issues,
        "allowed_chunk_count": len(allowed_ids),
        "citation_count": len(citations),
        "valid_citation_chunk_ids": valid_ids,
    }
