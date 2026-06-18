#!/usr/bin/env python3
"""
Final ranking step: combines BM25 scores with engineered features
to produce the final top 100 candidate ranking in the required CSV format.

Usage:
    python ranking.py [--input PATH] [--output PATH]
"""

import argparse
import csv
import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "candidates_featured.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "final_rankings.csv"

# Weights for the final score
WEIGHTS = {
    # Base relevance
    "bm25_score": 1.0,  # Base scale is ~0-200
    
    # Depth multipliers (raised caps to let top candidates differentiate)
    "retrieval_depth": 5.0,
    "evaluation_depth": 8.0,
    "production_depth": 8.0,
    
    # Career quality
    "avg_tenure_months": 0.2,
    "career_stability_score": 10.0,
    "current_title_score": 5.0,  # 0-5 scale * 5 = 0-25 pts
    "product_ratio": 20.0,       # Reward product company experience
    "unknown_ratio": -5.0,       # Reduced: unknown != bad (could be startups)
    
    # Platform signals (existing)
    "open_to_work_flag": 5.0,
    "willing_to_relocate": 5.0,
    "recruiter_response_rate": 20.0,
    "interview_completion_rate": 20.0,
    "offer_acceptance_rate": 20.0,
    "github_activity_score": 0.2, # 0-100 scale * 0.2 = 0-20 pts
    "saved_by_recruiters_30d": 5.0, # Will be log-scaled
    "search_appearance_30d": 3.0,   # Will be log-scaled
    
    # Location & domain
    "location_match": 3.0,       # Reduced to mild tie-breaker
    "domain_bonus_flag": 15.0,   # Bonus for marketplace/HR-tech experience
    
    # New signals from audit
    "profile_completeness_score": 0.1,  # 0-100 * 0.1 = 0-10 pts
    "profile_views_received_30d": 3.0,  # log-scaled, market validation
    "connection_count": 1.5,            # log-scaled, network proxy
    "endorsements_received": 2.0,       # log-scaled, peer validation
    "verified_composite": 3.0,          # 0-3 scale * 3 = 0-9 pts
    "skill_assessment_avg": 0.1,        # 0-100 * 0.1 = 0-10 pts
    "education_tier_score": 0.0,        # Removed per user feedback
    "skills_proficiency_score": 1.0,    # Weighted proficiency in relevant skills
    "company_size_score": 0.0,          # Removed per user feedback
    "country_match": 10.0,              # Reduced penalty for non-India
}

# Penalty features (0.0 to 1.0 scale)
PENALTIES = {
    "wrapper_penalty": 0.5,           # Up to 50% score reduction
    "cv_speech_robotics_penalty": 0.3, # Up to 30% score reduction
    "research_only_penalty": 0.4,      # Up to 40% score reduction
    "title_chaser_penalty": 0.2,       # Up to 20% score reduction
    "consulting_ratio": 0.3,           # Up to 30% score reduction
}

# Honeypot / Invalid Data rules (disqualification or severe penalty)
HONEYPOTS = [
    "education_timeline_valid", # 0 = suspicious
    "career_timeline_valid",    # 0 = suspicious
]


def score_row(row: pd.Series) -> float:
    score = 0.0
    
    # 1. Base Score
    score += row.get("bm25_score", 0.0) * WEIGHTS["bm25_score"]
    
    # 2. Additive signals — raised caps to capture top-end differentiation
    score += min(row.get("retrieval_depth", 0), 15) * WEIGHTS["retrieval_depth"]
    score += min(row.get("evaluation_depth", 0), 8) * WEIGHTS["evaluation_depth"]
    score += min(row.get("production_depth", 0), 10) * WEIGHTS["production_depth"]
    
    # YOE peaks at JD target (5-9 years)
    yoe = row.get("years_of_experience", 0)
    if 5 <= yoe <= 9:
        score += 30.0
    elif 3 <= yoe < 5:
        score += 5.0
    elif 9 < yoe <= 12:
        score += 10.0
    # else 0
    
    score += min(row.get("avg_tenure_months", 0), 48) * WEIGHTS["avg_tenure_months"]
    score += row.get("career_stability_score", 0) * WEIGHTS["career_stability_score"]
    score += row.get("current_title_score", 0) * WEIGHTS["current_title_score"]
    
    score += row.get("product_ratio", 0) * WEIGHTS["product_ratio"]
    score += row.get("unknown_ratio", 0) * WEIGHTS["unknown_ratio"]
    
    score += row.get("open_to_work_flag", 0) * WEIGHTS["open_to_work_flag"]
    score += row.get("willing_to_relocate", 0) * WEIGHTS["willing_to_relocate"]
    score += row.get("recruiter_response_rate", 0) * WEIGHTS["recruiter_response_rate"]
    score += row.get("interview_completion_rate", 0) * WEIGHTS["interview_completion_rate"]
    
    oar = row.get("offer_acceptance_rate", -1.0)
    if oar >= 0:
        score += oar * WEIGHTS["offer_acceptance_rate"]
        
    score += row.get("github_activity_score", 0) * WEIGHTS["github_activity_score"]
    
    # Log scaling for high-volume platform signals
    score += math.log1p(row.get("saved_by_recruiters_30d", 0)) * WEIGHTS["saved_by_recruiters_30d"]
    score += math.log1p(row.get("search_appearance_30d", 0)) * WEIGHTS["search_appearance_30d"]
    
    # Location tie breaker
    score += row.get("location_match", 0) * WEIGHTS["location_match"]
    
    # Domain knowledge bonus
    score += row.get("domain_bonus_flag", 0) * WEIGHTS["domain_bonus_flag"]
    
    # --- New signals from audit ---
    score += row.get("profile_completeness_score", 50) * WEIGHTS["profile_completeness_score"]
    score += math.log1p(row.get("profile_views_received_30d", 0)) * WEIGHTS["profile_views_received_30d"]
    score += math.log1p(row.get("connection_count", 0)) * WEIGHTS["connection_count"]
    score += math.log1p(row.get("endorsements_received", 0)) * WEIGHTS["endorsements_received"]
    score += row.get("verified_composite", 0) * WEIGHTS["verified_composite"]
    score += row.get("skill_assessment_avg", 0) * WEIGHTS["skill_assessment_avg"]
    score += row.get("education_tier_score", 0) * WEIGHTS["education_tier_score"]
    score += row.get("skills_proficiency_score", 0) * WEIGHTS["skills_proficiency_score"]
    score += row.get("company_size_score", 0) * WEIGHTS["company_size_score"]
    score += row.get("country_match", 0) * WEIGHTS["country_match"]
    
    # Response time: prefer faster responders (0 to -10 pts)
    response_time = row.get("avg_response_time_hours", 100)
    score -= min(response_time / 200.0, 1.0) * 10.0
    
    # Base score must be positive before multipliers
    score = max(score, 10.0)
    
    # 3. Multiplicative Penalties (0.0 to 1.0 scale, where 1.0 is max penalty)
    penalty_multiplier = 1.0
    for p_name, max_penalty in PENALTIES.items():
        p_val = row.get(p_name, 0.0)
        if p_name == "consulting_ratio" and p_val <= 0.8:
            continue # Only penalize heavily consulting candidates
        # Reduce multiplier by (p_val * max_penalty)
        penalty_multiplier *= (1.0 - (p_val * max_penalty))
    
    score *= penalty_multiplier
    
    # 3.5. Logistical & Behavioral Hard Constraints (Multiplicative)
    
    # [P1 FIX] Notice period multiplier — softened for 90+ day notice
    notice = row.get("notice_period_days", 90)
    if notice <= 30:
        score *= 1.0
    elif notice <= 60:
        score *= 0.85  # Was 0.8 — slightly softened
    elif notice <= 90:
        score *= 0.65  # Was 0.5 — significantly softened
    else:
        score *= 0.45  # Was 0.3 — softened to not obliterate strong candidates
        
    # Location and Relocation Multiplier
    country_match = row.get("country_match", 0)
    location_match = row.get("location_match", 0)
    relocate = row.get("willing_to_relocate", 0)
    
    if country_match == 0:
        if relocate == 0:
            score *= 0.1  # Out of country, won't relocate -> unhirable
        else:
            score *= 0.6  # Out of country, willing -> visa risk
    else:
        if location_match == 0 and relocate == 0:
            score *= 0.4  # Was 0.2 — softened; in India, may still relocate for right role
        elif location_match == 1:
            score *= 1.15  # Was 1.2 — slight location bonus, reduced to avoid over-weighting
            
    # Ghost Candidate Multiplier
    days_active = row.get("days_since_last_active", 365)
    rrr = row.get("recruiter_response_rate", 0.0)
    open_to_work = row.get("open_to_work_flag", 0)
    
    if open_to_work == 0 and days_active > 90:
        score *= 0.4
        
    # [P1 FIX] Recruiter response rate — softened from 0.4 to 0.6
    if rrr < 0.25:
        score *= 0.6  # Was 0.4 — technical fit matters more than response rate
        
    # [P1 FIX] External Validation — softened and broadened
    # JD says "papers, talks, open-source" not just GitHub
    github = row.get("github_activity_score", 0.0)
    yoe = row.get("years_of_experience", 0.0)
    endorsements = row.get("endorsements_received", 0)
    if github <= 0.0 and endorsements <= 0 and yoe >= 5.0:
        score *= 0.6  # Was 0.3 — softened; GitHub alone shouldn't be a death sentence
    
    # ===============================================================
    # [P0 FIX] Core Technical Fit Gate — Retrieval + Evaluation
    # The JD is crystal clear: the role owns "ranking, retrieval, and
    # matching systems". Candidates with ZERO evidence of both are
    # fundamentally unqualified regardless of behavioral signals.
    # ===============================================================
    ret_depth = row.get("retrieval_depth", 0)
    eval_depth = row.get("evaluation_depth", 0)
    
    if ret_depth == 0 and eval_depth == 0:
        score *= 0.25  # Severe: no evidence of core competency at all
    elif ret_depth + eval_depth <= 1:
        score *= 0.5   # Marginal: barely any evidence

    # [P1 FIX] Retrieval depth scaling — continuous reward for deeper experience
    # Candidates with ret_depth 3+ get full credit; below that, partial
    ret_scale = min(ret_depth / 3.0, 1.0)
    ret_scale = max(ret_scale, 0.4)  # Floor at 0.4 (don't double-penalize with gate above)
    score *= ret_scale
    
    # ===============================================================
    # [P0 FIX] Consulting Disqualifier — per JD explicit rules
    # "People who have only worked at consulting firms ... in their
    # entire career" is a disqualifier.
    # ===============================================================
    consulting_ratio = row.get("consulting_ratio", 0.0)
    currently_consulting = row.get("currently_at_consulting", 0)
    
    if consulting_ratio >= 1.0:
        # 100% consulting career = explicit JD disqualifier
        score *= 0.05
    elif currently_consulting == 1 and consulting_ratio > 0.5:
        # Currently at consulting firm + majority consulting career
        # JD says "If you're currently at one of these companies but
        # have prior product-company experience, that's fine" — this
        # catches those WITHOUT prior product experience
        score *= 0.2
    elif currently_consulting == 1:
        # Currently at consulting but with some product experience
        score *= 0.6
    
    # [P2 FIX] Currently-at-consulting + no ML title = extra penalty
    if currently_consulting == 1:
        title_score = row.get("current_title_score", 0)
        if title_score < 4:  # Not ML/AI/Search title
            score *= 0.5
    
    # 4. Honeypots / Consistency
    for hp in HONEYPOTS:
        if row.get(hp, 1) == 0:
            score *= 0.1  # Severe 90% penalty for invalid timelines
            
    experience_consistency = row.get("experience_consistency", 1.0)
    if experience_consistency < 0.5:
        score *= 0.5  # 50% penalty if claimed years don't match actual
        
    buzzword = row.get("buzzword_density", 0.0)
    if buzzword > 2.0: # 99th percentile
        score *= 0.8  # 20% penalty for keyword stuffing
        
    # False positive detector: If they have retrieval terms but ratio is tiny,
    # it means they listed it in skills but didn't write about it in their career.
    all_ret = row.get("retrieval_depth", 0)
    rer = row.get("retrieval_evidence_ratio", 1.0)
    if all_ret > 2 and rer < 0.2:
        score *= 0.7  # 30% penalty for likely skill-stuffers

    # [P0 FIX] Summary-vs-YOE mismatch honeypot detector
    summary_mismatch = row.get("summary_yoe_mismatch", 0.0)
    if summary_mismatch >= 1.0:
        score *= 0.0  # Severe: claims 2x+ what summary says — completely disqualify
    elif summary_mismatch > 0:
        score *= max(0.1, 1.0 - summary_mismatch)  # Gradual penalty

    # [P0 FIX] Expert skill inflation honeypot detector
    skill_inflation = row.get("expert_skill_inflation", 0.0)
    if skill_inflation >= 1.0:
        score *= 0.0  # Severe: expert in 8+ skills with 0 years — completely disqualify
    elif skill_inflation > 0:
        score *= 0.3  # Moderate: expert in 5+ skills with 0 years
        
    return round(score, 4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Reading candidates from {args.input} ...")
    df = pd.read_json(args.input, lines=True)
    print(f"Loaded {len(df):,} candidates.")
    
    print("Normalizing BM25 scores to 0-100 scale...")
    bm25_max = df["bm25_score"].max()
    bm25_min = df["bm25_score"].min()
    if bm25_max > bm25_min:
        df["bm25_score"] = 100.0 * (df["bm25_score"] - bm25_min) / (bm25_max - bm25_min)
    else:
        df["bm25_score"] = 50.0
    
    print("Calculating final scores...")
    df["final_score"] = df.apply(score_row, axis=1)
    
    # Sort by score descending, then candidate_id ascending (to break ties properly)
    df = df.sort_values(by=["final_score", "candidate_id"], ascending=[False, True])
    
    # Take top 100
    top_100 = df.head(100).copy()
    top_100["rank"] = range(1, 101)
    
    print("Generating reasoning strings...")
    def generate_reasoning(row):
        profile = row.get('profile', {}) or {}
        yoe = row.get('years_of_experience', 0)
        company = profile.get('current_company', 'their current role')
        rank_pos = row.get('rank', 50)  # Default to 50 if missing
        
        career = row.get('career_history', [])
        past_comps = []
        if isinstance(career, list):
            for role in career:
                if isinstance(role, dict):
                    c = role.get('company')
                    if c and c.lower() != company.lower() and c not in past_comps:
                        past_comps.append(c)
        
        comp_str = f"at {company}" if company != 'their current role' else "in their current role"
        if past_comps:
            comp_str += f" (previously {past_comps[0]})"
            
        skills = row.get('skills', [])
        rel_skills = []
        target_skills = {"python", "pytorch", "elasticsearch", "faiss", "tensorflow", "kubernetes", "search", "ranking", "machine learning", "nlp", "llms", "opensearch", "pinecone", "milvus"}
        if isinstance(skills, list):
            for s in skills:
                if isinstance(s, dict) and s.get('name'):
                    name = s['name']
                    if name.lower() in target_skills:
                        rel_skills.append(name)
        if not rel_skills and isinstance(skills, list):
            rel_skills = [s.get('name') for s in skills if isinstance(s, dict) and s.get('name')]
            
        skill_str = f", leveraging {', '.join(rel_skills[:3])}" if rel_skills else ""
        
        import hashlib
        import random
        h = int(hashlib.md5(row['candidate_id'].encode('utf-8')).hexdigest(), 16)
        rng = random.Random(h)
        
        # --- Extracted Points (text, base_magnitude, is_weakness) ---
        points = []
        
        # Strengths
        if row.get('retrieval_depth', 0) > 4:
            points.append(("deep production retrieval/ranking expertise", 50, False))
        elif row.get('retrieval_depth', 0) > 1:
            points.append(("solid retrieval fundamentals", 30, False))
            
        if row.get('evaluation_depth', 0) > 1:
            points.append(("hands-on evaluation experience (NDCG/MRR)", 45, False))
        elif row.get('evaluation_depth', 0) > 0:
            points.append(("basic search evaluation knowledge", 20, False))
            
        if row.get('production_depth', 0) > 4:
            points.append(("proven ML deployment scale", 35, False))
            
        if row.get('domain_bonus_flag', 0) > 0:
            points.append(("relevant HR-tech/marketplace domain experience", 30, False))
            
        if row.get('education_tier_score', 0) >= 4:
            points.append(("Tier-1 education", 20, False))
            
        if row.get('notice_period_days', 90) <= 15:
            points.append(("immediate availability", 25, False))
            
        if row.get('recruiter_response_rate', 0) > 0.85:
            points.append(("excellent recruiter responsiveness", 15, False))

        # Weaknesses (Mapped to pure noun phrases for flawless grammar)
        weakness_multiplier = rank_pos / 50.0  # Scales from ~0.02 (Rank 1) to 2.0 (Rank 100)
        
        if row.get('evaluation_depth', 0) == 0:
            points.append(("a lack of hands-on ranking evaluation metrics", 40 * weakness_multiplier, True))
            
        sig = row.get('redrob_signals', {})
        if sig.get('github_activity_score', 0) == 0 and sig.get('endorsements_received', 0) == 0:
            points.append(("zero external technical validation (GitHub/Endorsements)", 35 * weakness_multiplier, True))
            
        if row.get('wrapper_penalty', 0) > 0.2:
            points.append(("heavy reliance on API wrappers rather than core ML", 30 * weakness_multiplier, True))
            
        if row.get('consulting_ratio', 0) > 0.5:
            points.append(("a consulting-heavy career path", 25 * weakness_multiplier, True))
            
        if row.get('notice_period_days', 0) >= 90:
            points.append(("a long 90+ day notice period", 20 * weakness_multiplier, True))
            
        if yoe < 5.0:
            points.append(("a slight lack of overall experience for the target level", 20 * weakness_multiplier, True))

        # Sort by magnitude (highest absolute magnitude first)
        points.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 5
        top_points = points[:5]
        
        # Separate into strengths and weaknesses
        selected_strengths = [p[0] for p in top_points if not p[2]]
        selected_weaknesses = [p[0] for p in top_points if p[2]]
        
        # --- CFG Generation ---
        comp_base = f"{company}" if company != 'their current role' else "their current role"
        comp_prev = f" (previously {past_comps[0]})" if past_comps else ""
        
        # 1. Company Phrases
        companies = [
            f"at {comp_base}{comp_prev}",
            f"working at {comp_base}{comp_prev}",
            f"based out of {comp_base}{comp_prev}",
            f"from {comp_base}{comp_prev}"
        ]
        comp = rng.choice(companies)
        
        # 2. Skill Phrases
        skills_str = ', '.join(rel_skills[:3]) if rel_skills else "core software engineering"
        skill_phrases = [
            f"highly proficient in {skills_str}",
            f"specialized in {skills_str}",
            f"experienced with {skills_str}",
            f"skilled in leveraging {skills_str}",
            f"focused on {skills_str}"
        ]
        skill = rng.choice(skill_phrases)
        
        # 3. Sentence 1: YoE + Comp + Skill
        s1_options = [
            f"Bringing {yoe:.1f} years of experience {comp}, this candidate is {skill}.",
            f"With a {yoe:.1f}-year track record {comp}, they are {skill}.",
            f"This engineer offers {yoe:.1f} years of experience {comp}, and is {skill}.",
            f"Having spent {yoe:.1f} years {comp}, they are {skill}.",
            f"A seasoned engineer with {yoe:.1f} years {comp}, their expertise centers on {skills_str}."
        ]
        s1 = rng.choice(s1_options)
        
        # 4. Construct Sentence 2 (Strengths)
        if selected_strengths:
            strengths_str = ', '.join(selected_strengths)
            s2_options = [
                f"Notable strengths include {strengths_str}.",
                f"Key advantages are {strengths_str}.",
                f"The candidate's profile highlights {strengths_str}.",
                f"They offer standout qualities such as {strengths_str}.",
                f"Proven expertise in {strengths_str} makes them a strong fit."
            ]
            s2 = rng.choice(s2_options)
        else:
            s2 = "The candidate shows solid baseline technical foundations."
            
        # 5. Construct Sentence 3 (Weaknesses)
        s3 = ""
        if selected_weaknesses:
            weaknesses_str = ', '.join(selected_weaknesses)
            if len(selected_weaknesses) == 1:
                s3_options = [
                    f"However, the profile indicates {weaknesses_str}.",
                    f"A point of concern is {weaknesses_str}.",
                    f"It should be noted that the candidate has {weaknesses_str}.",
                    f"One minor drawback: {weaknesses_str}.",
                    f"On the downside, their background features {weaknesses_str}."
                ]
            else:
                s3_options = [
                    f"However, critical gaps include: {weaknesses_str}.",
                    f"Significant concerns exist, namely {weaknesses_str}.",
                    f"Points of friction include {weaknesses_str}.",
                    f"We must weigh these against notable gaps: {weaknesses_str}.",
                    f"Conversely, the profile shows weaknesses in: {weaknesses_str}."
                ]
            s3 = rng.choice(s3_options)
            
        res = f"{s1} {s2} {s3}".strip()
        return res.replace(" ,", ",").replace("  ", " ").strip()
        
    top_100["reasoning"] = top_100.apply(generate_reasoning, axis=1)
    
    # Write to CSV
    print(f"Writing top 100 to {args.output} ...")
    output_cols = ["candidate_id", "rank", "final_score", "reasoning"]
    
    # Rename final_score to score for output
    out_df = top_100.rename(columns={"final_score": "score"})
    out_df = out_df[["candidate_id", "rank", "score", "reasoning"]]
    
    out_df.to_csv(args.output, index=False)
    print("Done!")
    
    print("\nValidating submission...")
    import sys
    sys.path.append(str(PROJECT_ROOT / "scripts"))
    from validate_submission import validate_submission
    errors = validate_submission(args.output)
    if errors:
        print(f"Validation FAILED with {len(errors)} errors:")
        for e in errors:
            print(f"- {e}")
    else:
        print("Validation PASSED. Ready for submission.")


if __name__ == "__main__":
    main()
