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
        score += 15.0
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
    
    # Recency: mild decay based on days_since_last_active (0 to -20 pts)
    days = row.get("days_since_last_active", 365)
    score -= min(days / 365.0, 1.0) * 20.0
    
    # Notice period: non-linear penalty matching JD intent
    # JD: "We'd love sub-30-day notice. We can buy out up to 30 days.
    #      30+ day notice candidates are still in scope but the bar gets higher."
    notice = row.get("notice_period_days", 90)
    if notice <= 30:
        score -= 0.0   # No penalty — ideal per JD
    elif notice <= 60:
        score -= 5.0   # Mild — "still in scope"
    elif notice <= 90:
        score -= 12.0  # Significant — "bar gets higher"
    else:
        score -= 18.0  # Heavy — well beyond buyout range
    
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
    
    print("Calculating final scores...")
    df["final_score"] = df.apply(score_row, axis=1)
    
    # Sort by score descending, then candidate_id ascending (to break ties properly)
    df = df.sort_values(by=["final_score", "candidate_id"], ascending=[False, True])
    
    # Take top 100
    top_100 = df.head(100).copy()
    top_100["rank"] = range(1, 101)
    
    print("Generating reasoning strings...")
    def generate_reasoning(row):
        parts = []
        
        # 1. Core intro with company context
        yoe = row['years_of_experience']
        company = ""
        profile = row.get('profile')
        if isinstance(profile, dict):
            company = profile.get('current_company', '')
        
        if company:
            parts.append(f"{yoe:.1f} YoE, currently at {company}.")
        else:
            parts.append(f"{yoe:.1f} YoE candidate.")
        
        # 2. Depth details — specific to what the JD asks for
        prod = row['production_depth']
        ret = row['retrieval_depth']
        ev = row['evaluation_depth']
        if prod > 3 and ret > 3:
            parts.append("Has built and deployed production retrieval/ranking systems with evidence of scale.")
        elif ret > 3:
            parts.append("Strong evidence of search and retrieval system implementation.")
        elif prod > 3:
            parts.append("Demonstrated production ML engineering with deployment experience.")
        else:
            parts.append("Solid engineering foundations with relevant technical exposure.")
            
        if ev > 1:
            parts.append("Shows hands-on evaluation experience (NDCG/MRR/A-B testing).")
            
        # 3. Company quality
        pr = row['product_ratio']
        cr = row['consulting_ratio']
        if pr > 0.7:
            parts.append("Predominantly product-company career.")
        elif pr > 0.4:
            parts.append("Mix of product and other company experience.")
        elif cr > 0.8:
            parts.append("Primarily consulting background — higher risk per JD.")
            
        # 4. Education (if noteworthy)
        edu_tier = row.get('education_tier_score', 0)
        if edu_tier >= 4:
            # Try to get institution name
            education = row.get('education')
            if isinstance(education, list):
                for edu in education:
                    if isinstance(edu, dict) and edu.get('tier') == 'tier_1':
                        inst = edu.get('institution', '')
                        degree = edu.get('degree', '')
                        if inst:
                            parts.append(f"Tier-1 education ({degree} from {inst}).")
                            break
        
        # 5. Domain experience
        if row.get('domain_bonus_flag', 0):
            parts.append("Has relevant HR-tech or marketplace domain experience.")
            
        # 6. Platform engagement signals — specific and grammatically correct
        platform_signals = []
        if row['recruiter_response_rate'] > 0.8:
            parts.append(f"Highly responsive to recruiters ({row['recruiter_response_rate']:.0%} response rate).")
        oar = row.get('offer_acceptance_rate', -1)
        if oar > 0.8:
            parts.append(f"Strong offer acceptance track record ({oar:.0%}).")
        if row.get('willing_to_relocate', 0):
            platform_signals.append("open to relocation")
        if row.get('open_to_work_flag', 0):
            platform_signals.append("actively exploring opportunities")
        if platform_signals:
            parts.append(f"Currently {' and '.join(platform_signals)}.")
            
        # 7. Notice period (JD cares about this)
        notice = row.get('notice_period_days', 90)
        if notice <= 30:
            parts.append(f"Available within {notice} days — within buyout range.")
        elif notice > 90:
            parts.append(f"Note: {notice}-day notice period.")
            
        # 8. Location context
        country_match = row.get('country_match', 1.0)
        if country_match < 1.0:
            loc = ""
            if isinstance(profile, dict):
                loc = profile.get('location', 'outside India')
            parts.append(f"Based in {loc}; visa consideration needed.")
        elif row.get('location_match', 0):
            if isinstance(profile, dict):
                loc = profile.get('location', '')
                parts.append(f"Based in {loc} — preferred JD location.")
            
        return " ".join(parts)
        
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
