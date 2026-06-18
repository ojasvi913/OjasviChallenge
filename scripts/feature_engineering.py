#!/usr/bin/env python3
"""
Feature engineering for top BM25 candidates.

Reads BM25-scored candidates, takes the top N, and engineers features
across four groups: core, secondary, penalty, and honeypot.

Usage:
    python feature_engineering.py [--input PATH] [--output PATH] [--top-n 600]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from feature_config import (
    RETRIEVAL_DEPTH_TERMS,
    EVALUATION_DEPTH_TERMS,
    PRODUCTION_DEPTH_TERMS,
    CONSULTING_FIRMS,
    CONSULTING_INDUSTRY_KEYWORDS,
    PRODUCT_COMPANIES,
    PRODUCT_INDUSTRY_KEYWORDS,
    WRAPPER_TERMS,
    DEEPER_ML_TERMS,
    DEEPER_ML_TERMS_SECONDARY,
    CV_SPEECH_ROBOTICS_TERMS,
    NLP_IR_TERMS,
    RESEARCH_TERMS,
    PRODUCTION_EVIDENCE_TERMS,
    TITLE_SCORE_MAP,
    BUZZWORD_TERMS,
    SENIORITY_LEVELS,
    DOMAIN_KEYWORDS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "candidates_bm25_scored.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "candidates_featured.jsonl"
DEFAULT_TOP_N = 600
REFERENCE_DATE = date.today()


# ---------------------------------------------------------------------------
# Text helpers (shared across feature groups)
# ---------------------------------------------------------------------------

_NON_ALPHA = re.compile(r"[^a-z0-9/\-\s]")


def _clean(value: object) -> str:
    """Lowercase and strip non-alphanumeric characters."""
    if not isinstance(value, str):
        return ""
    return _NON_ALPHA.sub(" ", value.strip().lower())


def _count_terms(text: str, terms: list[str]) -> int:
    """Count unique vocabulary terms present in text (word-boundary aware)."""
    count = 0
    for term in terms:
        pattern = r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)"
        if re.search(pattern, text):
            count += 1
    return count


# --- Row-level accessors ---


def _get_signal(row: pd.Series, key: str, default=None):
    """Safely access a redrob_signals field."""
    signals = row.get("redrob_signals")
    if not isinstance(signals, dict):
        return default
    return signals.get(key, default)


def _get_profile(row: pd.Series, key: str, default=None):
    """Safely access a profile field."""
    profile = row.get("profile")
    if not isinstance(profile, dict):
        return default
    return profile.get(key, default)


def _parse_date(value: object) -> date | None:
    """Parse an ISO date string, returning None on failure."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


# --- Text extraction (computed once, reused by many features) ---


def _build_career_text(row: pd.Series) -> str:
    """Concatenate career description + title text for one candidate."""
    career = row.get("career_history")
    if not isinstance(career, list):
        return ""
    parts: list[str] = []
    for role in career:
        if not isinstance(role, dict):
            continue
        for key in ("description", "title"):
            text = _clean(role.get(key))
            if text:
                parts.append(text)
    return " ".join(parts)


def _build_all_text(row: pd.Series) -> str:
    """Concatenate career + profile + skills text for one candidate."""
    parts = [_build_career_text(row)]
    profile = row.get("profile")
    if isinstance(profile, dict):
        for key in ("summary", "headline", "current_title"):
            val = _clean(profile.get(key))
            if val:
                parts.append(val)
    skills = row.get("skills")
    if isinstance(skills, list):
        for skill in skills:
            if isinstance(skill, dict):
                name = _clean(skill.get("name"))
                if name:
                    parts.append(name)
    return " ".join(parts)


def _get_seniority_level(title: str) -> int:
    """Map a title string to a seniority level (0-6)."""
    best = 2  # default mid-level
    for keyword, level in SENIORITY_LEVELS.items():
        if keyword in title:
            best = max(best, level)
    return best


# ---------------------------------------------------------------------------
# Precompute shared columns (avoids recomputing text for every feature)
# ---------------------------------------------------------------------------


def _precompute_text_columns(candidates: pd.DataFrame) -> pd.DataFrame:
    result = candidates.copy()
    result["_career_text"] = result.apply(_build_career_text, axis=1)
    result["_all_text"] = result.apply(_build_all_text, axis=1)
    return result


# ===================================================================
# CORE FEATURES
# ===================================================================


def add_depth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add retrieval_depth, evaluation_depth, production_depth."""
    result = df.copy()
    ct = result["_career_text"]
    result["retrieval_depth"] = ct.apply(
        lambda t: _count_terms(t, RETRIEVAL_DEPTH_TERMS)
    )
    result["evaluation_depth"] = ct.apply(
        lambda t: _count_terms(t, EVALUATION_DEPTH_TERMS)
    )
    result["production_depth"] = ct.apply(
        lambda t: _count_terms(t, PRODUCTION_DEPTH_TERMS)
    )
    return result


def add_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features extracted directly from redrob_signals."""
    result = df.copy()

    result["open_to_work_flag"] = result.apply(
        lambda r: int(bool(_get_signal(r, "open_to_work_flag", False))), axis=1,
    )
    result["days_since_last_active"] = result.apply(
        _compute_days_since_active, axis=1,
    )
    result["recruiter_response_rate"] = result.apply(
        lambda r: float(_get_signal(r, "recruiter_response_rate", 0.0)), axis=1,
    )
    result["interview_completion_rate"] = result.apply(
        lambda r: float(_get_signal(r, "interview_completion_rate", 0.0)), axis=1,
    )
    result["notice_period_days"] = result.apply(
        lambda r: int(_get_signal(r, "notice_period_days", 180)), axis=1,
    )
    result["willing_to_relocate"] = result.apply(
        lambda r: int(bool(_get_signal(r, "willing_to_relocate", False))), axis=1,
    )
    result["location_match"] = result.apply(_compute_location_match, axis=1)

    # --- Previously unused signals ---
    result["profile_completeness_score"] = result.apply(
        lambda r: float(_get_signal(r, "profile_completeness_score", 50.0)), axis=1,
    )
    result["avg_response_time_hours"] = result.apply(
        lambda r: float(_get_signal(r, "avg_response_time_hours", 100.0)), axis=1,
    )
    result["applications_submitted_30d"] = result.apply(
        lambda r: int(_get_signal(r, "applications_submitted_30d", 0)), axis=1,
    )
    result["profile_views_received_30d"] = result.apply(
        lambda r: int(_get_signal(r, "profile_views_received_30d", 0)), axis=1,
    )
    result["connection_count"] = result.apply(
        lambda r: int(_get_signal(r, "connection_count", 0)), axis=1,
    )
    result["endorsements_received"] = result.apply(
        lambda r: int(_get_signal(r, "endorsements_received", 0)), axis=1,
    )
    # Composite verification signal: 0-3 scale
    result["verified_composite"] = result.apply(_compute_verified_composite, axis=1)
    # Country match: India = preferred, outside India = penalized per JD
    result["country_match"] = result.apply(_compute_country_match, axis=1)
    return result


def _compute_verified_composite(row: pd.Series) -> int:
    """Count of verification signals: verified_email + verified_phone + linkedin_connected."""
    score = 0
    if _get_signal(row, "verified_email", False):
        score += 1
    if _get_signal(row, "verified_phone", False):
        score += 1
    if _get_signal(row, "linkedin_connected", False):
        score += 1
    return score


def _compute_country_match(row: pd.Series) -> float:
    """1.0 = India (preferred), 0.0 = outside India (no visa sponsorship per JD)."""
    country = _clean(_get_profile(row, "country", ""))
    if "india" in country:
        return 1.0
    return 0.0


def _compute_location_match(row: pd.Series) -> int:
    loc = _clean(_get_profile(row, "location", ""))
    for tz in ["pune", "noida", "delhi", "mumbai", "hyderabad", "bengaluru", "bangalore"]:
        if tz in loc:
            return 1
    return 0


def _compute_days_since_active(row: pd.Series) -> int:
    d = _parse_date(_get_signal(row, "last_active_date"))
    if d is None:
        return 365
    return max((REFERENCE_DATE - d).days, 0)


def add_company_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add product_ratio, consulting_ratio, and unknown_ratio."""
    result = df.copy()
    ratios = result.apply(_compute_company_ratios, axis=1, result_type="expand")
    result["product_ratio"] = ratios[0]
    result["consulting_ratio"] = ratios[1]
    result["unknown_ratio"] = ratios[2]
    return result


def _classify_company(company: str, industry: str) -> str:
    """Classify a career entry as 'consulting', 'product', or 'other'."""
    comp = company.lower() if isinstance(company, str) else ""
    ind = industry.lower() if isinstance(industry, str) else ""

    # Check consulting first
    for firm in CONSULTING_FIRMS:
        if firm in comp:
            return "consulting"
    for kw in CONSULTING_INDUSTRY_KEYWORDS:
        if kw in ind:
            return "consulting"

    # Check known product companies
    for firm in PRODUCT_COMPANIES:
        if firm in comp:
            return "product"
    for kw in PRODUCT_INDUSTRY_KEYWORDS:
        if kw in ind:
            return "product"

    return "other"


def _compute_company_ratios(row: pd.Series) -> tuple[float, float, float]:
    career = row.get("career_history")
    if not isinstance(career, list) or not career:
        return (0.0, 0.0, 1.0)

    total_months = 0
    consulting_months = 0
    product_months = 0
    for role in career:
        if not isinstance(role, dict):
            continue
        dur = role.get("duration_months", 0)
        if not isinstance(dur, (int, float)) or dur <= 0:
            continue
        total_months += dur
        label = _classify_company(role.get("company", ""), role.get("industry", ""))
        if label == "consulting":
            consulting_months += dur
        elif label == "product":
            product_months += dur

    if total_months == 0:
        return (0.0, 0.0, 1.0)

    p_ratio = round(product_months / total_months, 3)
    c_ratio = round(consulting_months / total_months, 3)
    u_ratio = round(1.0 - p_ratio - c_ratio, 3)
    return (p_ratio, c_ratio, u_ratio)


# ===================================================================
# SECONDARY FEATURES
# ===================================================================


def add_secondary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add years_of_experience, tenure, title score, and signal features."""
    result = df.copy()

    # years_of_experience
    result["years_of_experience"] = result.apply(
        lambda r: float(_get_profile(r, "years_of_experience", 0.0)), axis=1,
    )

    # avg_tenure_months + career_stability_score
    tenure = result.apply(_compute_tenure_features, axis=1, result_type="expand")
    result["avg_tenure_months"] = tenure[0]
    result["career_stability_score"] = tenure[1]

    # current_title_score
    result["current_title_score"] = result.apply(_compute_title_score, axis=1)

    # 6. Domain specific exposure
    result["domain_bonus_flag"] = result["_career_text"].apply(
        lambda t: 1 if _count_terms(t, DOMAIN_KEYWORDS) > 0 else 0
    )
    
    # 7. RedRob Signals
    result["github_activity_score"] = result.apply(
        lambda r: float(max(_get_signal(r, "github_activity_score", -1), 0)), axis=1,
    )
    result["saved_by_recruiters_30d"] = result.apply(
        lambda r: int(_get_signal(r, "saved_by_recruiters_30d", 0)), axis=1,
    )
    result["search_appearance_30d"] = result.apply(
        lambda r: int(_get_signal(r, "search_appearance_30d", 0)), axis=1,
    )
    
    def _parse_oar(r):
        val = _get_signal(r, "offer_acceptance_rate")
        if val is None or float(val) < 0:
            return -1.0
        return float(val)
        
    result["offer_acceptance_rate"] = result.apply(_parse_oar, axis=1)
    
    result["retrieval_evidence_ratio"] = result.apply(_compute_retrieval_ratio, axis=1)

    # 8. Education tier (best tier across all degrees)
    result["education_tier_score"] = result.apply(_compute_education_tier, axis=1)

    # 9. Skill assessment average (platform-verified competency)
    result["skill_assessment_avg"] = result.apply(_compute_skill_assessment_avg, axis=1)

    # 10. Skills depth: weighted proficiency in relevant skills
    result["skills_proficiency_score"] = result.apply(_compute_skills_proficiency, axis=1)

    # 11. Company size signal (larger = more likely scale experience)
    result["company_size_score"] = result.apply(_compute_company_size_score, axis=1)

    return result

def _compute_retrieval_ratio(row: pd.Series) -> float:
    career_ret = _count_terms(row["_career_text"], RETRIEVAL_DEPTH_TERMS)
    all_ret = _count_terms(row["_all_text"], RETRIEVAL_DEPTH_TERMS)
    if all_ret == 0:
        return 0.0
    return round(career_ret / all_ret, 3)


def _compute_education_tier(row: pd.Series) -> int:
    """Best (lowest-numbered) education tier. tier_1=4pts, tier_2=3, tier_3=2, tier_4=1, unknown=0."""
    education = row.get("education")
    if not isinstance(education, list) or not education:
        return 0
    tier_map = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}
    best = 0
    for edu in education:
        if isinstance(edu, dict):
            tier = edu.get("tier", "unknown")
            best = max(best, tier_map.get(tier, 0))
    return best


def _compute_skill_assessment_avg(row: pd.Series) -> float:
    """Average of platform skill assessment scores (0-100). 0 if none taken."""
    scores = _get_signal(row, "skill_assessment_scores")
    if not isinstance(scores, dict) or not scores:
        return 0.0
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    if not vals:
        return 0.0
    return round(sum(vals) / len(vals), 1)


_PROFICIENCY_MAP = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}

# Key skills from the JD that signal real depth
_RELEVANT_SKILL_NAMES = {
    "python", "pytorch", "tensorflow", "elasticsearch", "opensearch",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "xgboost", "lightgbm", "scikit-learn", "sklearn",
    "sentence-transformers", "sentence transformers", "bert", "transformers",
    "docker", "kubernetes", "airflow", "kafka", "spark",
    "nlp", "machine learning", "deep learning", "information retrieval",
    "recommendation systems", "search", "ranking",
}


def _compute_skills_proficiency(row: pd.Series) -> float:
    """Weighted score from skill proficiency levels in relevant skills."""
    skills = row.get("skills")
    if not isinstance(skills, list):
        return 0.0
    total = 0.0
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        name = (skill.get("name", "") or "").strip().lower()
        if name in _RELEVANT_SKILL_NAMES:
            prof = _PROFICIENCY_MAP.get(skill.get("proficiency", ""), 0)
            endorsements = skill.get("endorsements", 0) or 0
            # Weight: proficiency level + log bonus from endorsements
            total += prof + math.log1p(endorsements) * 0.3
    return round(total, 2)


_COMPANY_SIZE_MAP = {
    "10001+": 4, "5001-10000": 3, "1001-5000": 3,
    "501-1000": 2, "201-500": 2, "51-200": 1, "11-50": 1, "1-10": 0,
}


def _compute_company_size_score(row: pd.Series) -> int:
    """Score based on current company size. Larger = more likely scale experience."""
    size = _get_profile(row, "current_company_size", "")
    return _COMPANY_SIZE_MAP.get(size, 0)


def _compute_tenure_features(row: pd.Series) -> tuple[float, float]:
    career = row.get("career_history")
    if not isinstance(career, list) or not career:
        return (0.0, 0.0)

    durations: list[float] = []
    for role in career:
        if not isinstance(role, dict):
            continue
        d = role.get("duration_months", 0)
        if isinstance(d, (int, float)) and d > 0:
            durations.append(float(d))

    if not durations:
        return (0.0, 0.0)

    avg_tenure = sum(durations) / len(durations)

    # Career stability: 0–4 points per role based on duration
    stability = 0.0
    for d in durations:
        if d < 12:
            stability += 0
        elif d < 24:
            stability += 1
        elif d < 36:
            stability += 2
        elif d < 48:
            stability += 3
        else:
            stability += 4
    stability /= len(durations)

    return (round(avg_tenure, 1), round(stability, 2))


def _compute_title_score(row: pd.Series) -> int:
    title = _clean(_get_profile(row, "current_title", ""))
    if not title:
        return 0
    for score, terms in TITLE_SCORE_MAP:
        for term in terms:
            if term in title:
                return score
    return 0


# ===================================================================
# PENALTY FEATURES (0.0 = no penalty, up to 1.0 = max penalty)
# ===================================================================


def add_penalty_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add wrapper, CV/speech/robotics, research-only, and title-chaser penalties."""
    result = df.copy()
    result["wrapper_penalty"] = result.apply(_compute_wrapper_penalty, axis=1)
    result["cv_speech_robotics_penalty"] = result.apply(
        _compute_cv_speech_penalty, axis=1,
    )
    result["research_only_penalty"] = result.apply(
        _compute_research_penalty, axis=1,
    )
    result["title_chaser_penalty"] = result.apply(
        _compute_title_chaser_penalty, axis=1,
    )
    return result


def _compute_wrapper_penalty(row: pd.Series) -> float:
    """High when wrapper/API terms dominate relative to actual depth.

    wrapper_ratio = wrapper_terms / (deeper_ml + deeper_ml_secondary + retrieval_depth + production_depth + 1)
    Candidates with real depth naturally dilute the wrapper signal.
    """
    text = row["_all_text"]
    wrapper = _count_terms(text, WRAPPER_TERMS)
    if wrapper == 0:
        return 0.0
    deeper = _count_terms(text, DEEPER_ML_TERMS)
    deeper_secondary = _count_terms(text, DEEPER_ML_TERMS_SECONDARY)
    retrieval = row.get("retrieval_depth", 0)
    production = row.get("production_depth", 0)
    denominator = deeper + deeper_secondary + retrieval + production + 1
    ratio = wrapper / denominator
    return round(min(ratio, 1.0), 3)


def _compute_cv_speech_penalty(row: pd.Series) -> float:
    """High when CV/speech/robotics terms dominate over NLP/IR terms."""
    text = row["_all_text"]
    cv = _count_terms(text, CV_SPEECH_ROBOTICS_TERMS)
    nlp = _count_terms(text, NLP_IR_TERMS)
    if cv == 0:
        return 0.0
    if nlp == 0:
        return round(min(cv / 5.0, 1.0), 3)
    if cv / max(nlp, 1) > 2.0:
        return round(min((cv / nlp) * 0.3, 1.0), 3)
    return 0.0


def _compute_research_penalty(row: pd.Series) -> float:
    """Research-ONLY penalty: fires when research terms are high AND
    production_depth is low. Research itself is not penalised."""
    text = row["_career_text"]
    research = _count_terms(text, RESEARCH_TERMS)
    if research == 0:
        return 0.0
    production = row.get("production_depth", 0)
    # Only penalise when production evidence is thin
    if production >= 3:
        return 0.0  # Has real production depth — no penalty
    if production >= 1:
        # Some production evidence, mild penalty only if heavily research-skewed
        if research > production * 3:
            return round(min((research - production * 3) / 5.0, 0.5), 3)
        return 0.0
    # Zero production depth + research terms → penalty
    return round(min(research / 5.0, 1.0), 3)


def _compute_title_chaser_penalty(row: pd.Series) -> float:
    """High when career shows repeated short tenures with seniority jumps."""
    career = row.get("career_history")
    if not isinstance(career, list) or len(career) < 2:
        return 0.0

    dated_roles: list[tuple[str, str, float]] = []
    for role in career:
        if not isinstance(role, dict):
            continue
        start = role.get("start_date")
        title = _clean(role.get("title", ""))
        duration = role.get("duration_months", 0)
        if isinstance(start, str) and title:
            dated_roles.append((
                start,
                title,
                float(duration) if isinstance(duration, (int, float)) else 0.0,
            ))

    dated_roles.sort(key=lambda x: x[0])

    rapid_jumps = 0
    for i in range(len(dated_roles) - 1):
        _, title_a, dur_a = dated_roles[i]
        _, title_b, _ = dated_roles[i + 1]
        if dur_a >= 18:
            continue
        if _get_seniority_level(title_b) > _get_seniority_level(title_a):
            rapid_jumps += 1

    return round(min(rapid_jumps / 3.0, 1.0), 3)


# ===================================================================
# HONEYPOT FEATURES
# ===================================================================


def add_honeypot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add data-integrity and synthetic-profile detection features."""
    result = df.copy()
    result["education_timeline_valid"] = result.apply(
        _check_education_timeline, axis=1,
    )
    result["career_timeline_valid"] = result.apply(
        _check_career_timeline, axis=1,
    )
    result["experience_consistency"] = result.apply(
        _check_experience_consistency, axis=1,
    )
    result["title_progression_suspicious"] = result.apply(
        _check_title_progression, axis=1,
    )
    result["buzzword_density"] = result.apply(
        _compute_buzzword_density, axis=1,
    )
    # P0 Fix: Cross-check summary text stated years vs claimed YOE field
    result["summary_yoe_mismatch"] = result.apply(
        _check_summary_yoe_mismatch, axis=1,
    )
    # P0 Fix: Detect expert skill inflation (expert in many skills with 0 years_used)
    result["expert_skill_inflation"] = result.apply(
        _check_expert_skill_inflation, axis=1,
    )
    # P2 Fix: Flag candidates currently employed at consulting firms
    result["currently_at_consulting"] = result.apply(
        _check_currently_at_consulting, axis=1,
    )
    return result


def _check_education_timeline(row: pd.Series) -> int:
    """1 = valid, 0 = suspicious."""
    education = row.get("education")
    if not isinstance(education, list) or not education:
        return 1
    for edu in education:
        if not isinstance(edu, dict):
            continue
        start = edu.get("start_year")
        end = edu.get("end_year")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if end - start < 0 or end - start > 10:
            return 0
        if start < 1970 or end > 2030:
            return 0
    return 1


def _check_career_timeline(row: pd.Series) -> int:
    """1 = valid, 0 = suspicious."""
    career = row.get("career_history")
    if not isinstance(career, list) or not career:
        return 1
    for role in career:
        if not isinstance(role, dict):
            continue
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date"))
        if start and end and end < start:
            return 0
        dur = role.get("duration_months", 0)
        if isinstance(dur, (int, float)) and dur < 0:
            return 0
    return 1


def _check_experience_consistency(row: pd.Series) -> float:
    """0–1 ratio. Higher = claimed experience matches career history."""
    claimed = _get_profile(row, "years_of_experience", 0)
    if not isinstance(claimed, (int, float)) or claimed <= 0:
        return 0.5

    career = row.get("career_history")
    if not isinstance(career, list) or not career:
        return 0.5

    total_months = 0.0
    for role in career:
        if not isinstance(role, dict):
            continue
        d = role.get("duration_months", 0)
        if isinstance(d, (int, float)) and d > 0:
            total_months += d

    actual = total_months / 12.0
    if actual == 0:
        return 0.5

    return round(min(claimed, actual) / max(claimed, actual), 3)


def _check_title_progression(row: pd.Series) -> float:
    """0.0 = normal, up to 1.0 = suspicious (big seniority jumps in short time)."""
    career = row.get("career_history")
    if not isinstance(career, list) or len(career) < 3:
        return 0.0

    dated_roles: list[tuple[str, str, float]] = []
    for role in career:
        if not isinstance(role, dict):
            continue
        start = role.get("start_date")
        title = _clean(role.get("title", ""))
        dur = role.get("duration_months", 0)
        if isinstance(start, str) and title:
            dated_roles.append((
                start,
                title,
                float(dur) if isinstance(dur, (int, float)) else 0.0,
            ))

    dated_roles.sort(key=lambda x: x[0])

    suspicious = 0
    for i in range(len(dated_roles) - 1):
        _, title_a, dur_a = dated_roles[i]
        _, title_b, _ = dated_roles[i + 1]
        level_diff = _get_seniority_level(title_b) - _get_seniority_level(title_a)
        if level_diff >= 2 and dur_a < 24:
            suspicious += 1

    return round(min(suspicious / 2.0, 1.0), 3)


def _compute_buzzword_density(row: pd.Series) -> float:
    """Buzzword matches per 100 words. Higher = more suspicious."""
    text = row["_all_text"]
    if not text:
        return 0.0
    total_words = len(text.split())
    if total_words == 0:
        return 0.0
    buzzwords = _count_terms(text, BUZZWORD_TERMS)
    return round(buzzwords / total_words * 100, 3)


_YOE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\+\s*)?years?\s+(?:of\s+)?(?:hands?-?on\s+)?(?:experience|building|in|across)", re.IGNORECASE)


def _check_summary_yoe_mismatch(row: pd.Series) -> float:
    """Detect mismatch between claimed YOE field and years stated in summary text.

    Returns 0.0 (no mismatch) to 1.0 (severe mismatch).
    Honeypot profiles often inflate the structured YOE field but write a
    realistic number in their free-text summary.
    """
    claimed = _get_profile(row, "years_of_experience", 0)
    if not isinstance(claimed, (int, float)) or claimed <= 0:
        return 0.0

    summary = _get_profile(row, "summary", "")
    if not isinstance(summary, str) or not summary:
        return 0.0

    matches = _YOE_PATTERN.findall(summary)
    if not matches:
        return 0.0

    # Take the largest number mentioned (most likely the total experience)
    stated_years = max(float(m) for m in matches)

    if stated_years <= 0:
        return 0.0

    # If claimed is more than 1.5x what's stated in summary, suspicious
    ratio = claimed / stated_years
    if ratio > 2.0:
        return 1.0  # Severe: claims 2x+ what summary says
    elif ratio > 1.5:
        return round((ratio - 1.5) * 2.0, 3)  # Gradual: 1.5x-2.0x
    return 0.0


def _check_expert_skill_inflation(row: pd.Series) -> float:
    """Detect candidates claiming 'expert' proficiency in many skills with 0 years_used.

    Returns 0.0 (normal) to 1.0 (highly suspicious).
    Honeypot pattern: expert in 10 skills with 0 years experience in each.
    Note: years_used == -1 means 'not reported' (common in this dataset),
    so we only flag when years_used is explicitly 0.
    """
    skills = row.get("skills")
    if not isinstance(skills, list) or not skills:
        return 0.0

    expert_zero_count = 0
    expert_count = 0
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        prof = (skill.get("proficiency", "") or "").lower()
        if prof == "expert":
            expert_count += 1
            years_used = skill.get("years_used", -1)
            # Only count explicitly 0, not -1 (unknown/not reported)
            if isinstance(years_used, (int, float)) and years_used == 0:
                expert_zero_count += 1

    if expert_count >= 8 and expert_zero_count >= 6:
        return 1.0
    if expert_count >= 5 and expert_zero_count >= 4:
        return 0.5
    return 0.0


def _check_currently_at_consulting(row: pd.Series) -> int:
    """1 if the candidate is currently employed at a consulting/services firm.

    The JD explicitly disqualifies candidates whose *entire* career is at
    consulting firms, and is cautious about those currently there.
    """
    company = _clean(_get_profile(row, "current_company", ""))
    industry = _clean(_get_profile(row, "current_industry", ""))
    label = _classify_company(company, industry)
    return 1 if label == "consulting" else 0


# ===================================================================
# PIPELINE ORCHESTRATION
# ===================================================================

_INTERNAL_COLUMNS = ["_career_text", "_all_text"]


def engineer_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline on a candidate DataFrame."""
    candidates = _precompute_text_columns(candidates)
    candidates = add_depth_features(candidates)
    candidates = add_signal_features(candidates)
    candidates = add_company_features(candidates)
    candidates = add_secondary_features(candidates)
    candidates = add_penalty_features(candidates)
    candidates = add_honeypot_features(candidates)
    # Log the new feature count
    return candidates.drop(columns=_INTERNAL_COLUMNS, errors="ignore")


# ===================================================================
# CLI
# ===================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Engineer features for top BM25 candidates.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top BM25 candidates to feature. Default: {DEFAULT_TOP_N}",
    )
    return parser.parse_args()


# Columns from the raw candidate JSON that are not engineered features.
_RAW_COLUMNS = {
    "candidate_id", "profile", "career_history", "education",
    "skills", "certifications", "languages", "redrob_signals",
    "bm25_production_ml", "bm25_retrieval_ranking", "bm25_score",
}


def main() -> None:
    args = parse_args()

    print(f"Reading candidates from {args.input} ...")
    candidates = pd.read_json(args.input, lines=True)
    print(f"Loaded {len(candidates):,} candidates.")

    candidates = candidates.nlargest(args.top_n, "bm25_score").reset_index(drop=True)
    print(f"Selected top {len(candidates):,} by BM25 score.")

    print("Engineering features ...")
    featured = engineer_features(candidates)

    # Write output — use json.dumps per row for safe escaping
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in featured.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # Summary
    feature_cols = [c for c in featured.columns if c not in _RAW_COLUMNS]
    print(f"\nEngineered {len(feature_cols)} features for {len(featured):,} candidates.")
    print(f"\nFeature summary:")
    summary = featured[feature_cols].describe().T
    print(summary.to_string())
    print(f"\nWrote output to {args.output}")


if __name__ == "__main__":
    main()
