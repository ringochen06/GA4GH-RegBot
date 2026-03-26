from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.regbot.config import DEFAULT_LLM_MODEL
from src.regbot.grounding import allowed_chunk_ids, audit_citation_grounding


def _format_evidence(chunks: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    parts: List[str] = []
    used = 0
    for c in chunks:
        meta = c.get("metadata") or {}
        header = (
            f"[chunk_id={c['id']} source={meta.get('source','?')} "
            f"page={meta.get('page',0)} category={meta.get('category','?')}]"
        )
        body = c.get("text", "").strip()
        block = f"{header}\n{body}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts).strip()


def _fallback_report(
    consent: str,
    study_type: str,
    chunks: List[Dict[str, Any]],
    *,
    grounding_strict: bool = True,
) -> Dict[str, Any]:
    """Heuristic output when no LLM key is available; still ties claims to chunk ids."""
    consent_l = consent.lower()
    keywords = [
        "secondary use",
        "recontact",
        "withdraw",
        "data sharing",
        "international",
        "cloud",
        "identifiable",
        "pseudonym",
    ]
    missing = [k for k in keywords if k not in consent_l]
    status = "Partially Compliant" if len(missing) <= 3 else "Non-Compliant"
    citations = []
    for c in chunks[:5]:
        citations.append(
            {
                "chunk_id": c["id"],
                "reason": "Retrieved as potentially relevant policy context.",
            }
        )
    out = {
        "study_type": study_type,
        "status": status,
        "missing_elements": missing,
        "recommendations": [
            "Add explicit language on permitted secondary uses and any restrictions.",
            "Clarify withdrawal of consent and what happens to already-shared data.",
            "State whether data may be stored or processed outside the original jurisdiction.",
        ],
        "citations": citations,
        "notes": (
            "LLM analysis disabled (no OPENAI_API_KEY). "
            "This is a lightweight keyword gap check, not legal advice."
        ),
    }
    allow = allowed_chunk_ids(chunks)
    out["grounding"] = audit_citation_grounding(
        out,
        allow,
        strict_min_citations_vs_recommendations=grounding_strict,
    )
    out["grounding_attempts"] = 1
    return out


def analyze_compliance(
    consent_text: str,
    chunks: List[Dict[str, Any]],
    *,
    study_type: str,
    api_key: Optional[str],
    model: Optional[str] = None,
    max_grounding_retries: int = 1,
    grounding_strict: bool = True,
) -> Dict[str, Any]:
    if not chunks:
        return {
            "study_type": study_type,
            "status": "Unknown",
            "missing_elements": [],
            "recommendations": [
                "Ingest at least one GA4GH policy document into the local store before checking compliance."
            ],
            "citations": [],
            "notes": "No retrieved policy context.",
        }

    if not api_key:
        return _fallback_report(
            consent_text,
            study_type,
            chunks,
            grounding_strict=grounding_strict,
        )

    allow = allowed_chunk_ids(chunks)
    evidence = _format_evidence(chunks)
    model_name = model or DEFAULT_LLM_MODEL
    client = OpenAI(api_key=api_key)

    system = (
        "You are a research compliance assistant for genomic data sharing. "
        "You must only use the POLICY EXCERPTS block as evidence. "
        "If the excerpts do not contain enough information, say so explicitly. "
        "Every recommendation must have supporting citations. "
        "Each citations[].chunk_id MUST be copied exactly from a [chunk_id=...] header in POLICY EXCERPTS "
        "(no invented ids). "
        "Return JSON only, no markdown."
    )
    base_user = (
        f"STUDY_TYPE_GUESS: {study_type}\n\n"
        f"POLICY EXCERPTS:\n{evidence}\n\n"
        f"CONSENT_OR_DATA_USE_TEXT:\n{consent_text.strip()}\n\n"
        "Return a JSON object with keys: "
        "study_type (string), status (one of: Compliant, Partially Compliant, Non-Compliant, Unknown), "
        "missing_elements (array of strings), recommendations (array of strings), "
        "citations (array of objects with chunk_id, reason), "
        "notes (optional string). "
        "Use at least as many citations as recommendations; each citation.chunk_id must be in the allow-list "
        "implied by the excerpt headers."
    )

    suffix = ""
    attempts = 0
    max_attempts = max(1, max_grounding_retries + 1)
    out: Dict[str, Any] = {}
    while attempts < max_attempts:
        attempts += 1
        user = base_user + suffix
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", raw)
            data = json.loads(m.group(0)) if m else {}

        out = {
            "study_type": str(data.get("study_type", study_type)),
            "status": str(data.get("status", "Unknown")),
            "missing_elements": list(data.get("missing_elements") or []),
            "recommendations": list(data.get("recommendations") or []),
            "citations": list(data.get("citations") or []),
        }
        if data.get("notes"):
            out["notes"] = str(data["notes"])
        out["model"] = model_name

        audit = audit_citation_grounding(
            out,
            allow,
            strict_min_citations_vs_recommendations=grounding_strict,
        )
        out["grounding"] = audit
        out["grounding_attempts"] = attempts
        if audit["ok"] or attempts >= max_attempts:
            if not audit["ok"]:
                note = (
                    " Model output failed programmatic citation grounding checks after retries; "
                    "treat recommendations as ungrounded."
                )
                out["notes"] = (out.get("notes") or "") + note
            return out

        suffix = (
            "\n\nGROUNDING_FIX_REQUIRED:\n"
            + "\n".join(f"- {issue}" for issue in audit["issues"])
            + "\nAllowed chunk_id values (copy exactly):\n"
            + json.dumps(sorted(allow))
            + "\nReturn the full JSON object again with corrected citations."
        )

    return out
