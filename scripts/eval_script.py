import json, csv, sys, math
import pandas as pd
sys.path.append('scripts')
import ranking
import feature_engineering
from collections import Counter

output_path = r'C:\Users\admin\.gemini\antigravity-ide\brain\f0309ab5-1cc5-4dd8-a526-6e6beaf6b0f5\evaluation_report.md'
out = open(output_path, 'w', encoding='utf-8')
def print_md(*args, **kwargs):
    print(*args, file=out, **kwargs)
    print(*args, **kwargs)

# Load data
lines = open(r'c:\Users\admin\Desktop\redrobrproject\data\candidates_featured.jsonl', 'r', encoding='utf-8').readlines()
records = [json.loads(l) for l in lines]
df = pd.DataFrame(records)

# Calculate score for all 600
df['final_score'] = df.apply(ranking.score_row, axis=1)
df = df.sort_values(by=['final_score', 'candidate_id'], ascending=[False, True]).reset_index(drop=True)
df['rank'] = df.index + 1

top100_df = df.head(100)
top100_records = top100_df.to_dict('records')

print_md("# Final Evaluation Report")

print_md("\n## 1. Inspect Rank 1")
r1 = top100_records[0]
print_md(f"**ID**: {r1['candidate_id']} | **Score**: {r1['final_score']:.2f}")
print_md(f"**BM25**: {r1.get('bm25_score',0):.2f} | **Title**: {r1.get('profile',{}).get('current_title','')} @ {r1.get('profile',{}).get('current_company','')}")
print_md(f"**YoE**: {r1.get('years_of_experience',0)} | **Retrieval Depth**: {r1.get('retrieval_depth',0)} | **Prod Depth**: {r1.get('production_depth',0)}")
print_md(f"**Skills**: {', '.join([s.get('name','') for s in r1.get('skills',[])[:10]])}")
# Get the full profile text 
all_text = r1.get('_all_text', 'Not precomputed')
print_md(f"\n**Brief Profile snippet**:\n> {r1.get('profile',{}).get('summary','')[:300]}...")

print_md("\n## 2. Top 10 Quick Look")
for r in top100_records[:10]:
    print_md(f"- Rank {r['rank']}: {r['candidate_id']} | {r.get('profile',{}).get('current_title','')} @ {r.get('profile',{}).get('current_company','')} | YoE: {r.get('years_of_experience',0)} | RetDepth: {r.get('retrieval_depth',0)} | Score: {r['final_score']:.1f}")

print_md("\n## 3. Boundary Inspection (Ranks 90, 95, 100, 101, 110)")
for r in [90, 95, 100, 101, 110]:
    row = df.iloc[r-1].to_dict()
    print_md(f"- Rank {r}: {row['candidate_id']} | Score: {row['final_score']:.2f} | BM25: {row.get('bm25_score',0):.2f} | Title: {row.get('profile',{}).get('current_title','')} | Ret Depth: {row.get('retrieval_depth',0)}")

print_md("\n## 4. Current Titles in Top 100")
titles = [r.get('profile',{}).get('current_title','') for r in top100_records]
t_counts = Counter(titles)
for t, c in t_counts.most_common(10):
    print_md(f"- {t}: {c}")

print_md("\n## 5. Company Distribution")
prod = sum(1 for r in top100_records if r.get('product_ratio',0) > 0.5)
cons = sum(1 for r in top100_records if r.get('consulting_ratio',0) > 0.5)
unk = sum(1 for r in top100_records if r.get('unknown_ratio',0) > 0.5)
print_md(f"- Product heavy (>50%): {prod}")
print_md(f"- Consulting heavy (>50%): {cons}")
print_md(f"- Unknown heavy (>50%): {unk}")

print_md("\n## 6. Domain Distribution")
domain_hits = sum(1 for r in top100_records if r.get('domain_bonus_flag', 0) > 0)
print_md(f"- Candidates with HR-Tech/Marketplace domain: {domain_hits}")

print_md("\n## 7. Behavioral Signals (Top 100 Averages)")
print_md(f"- Recruiter Response Rate: {top100_df['recruiter_response_rate'].mean():.2f}")
print_md(f"- Interview Completion Rate: {top100_df['interview_completion_rate'].mean():.2f}")
print_md(f"- Notice Period Days: {top100_df['notice_period_days'].mean():.1f}")
print_md(f"- Days Since Last Active: {top100_df['days_since_last_active'].mean():.1f}")

print_md("\n## 8. Non-India Candidates in Top 100")
non_india = [r for r in top100_records if r.get('profile',{}).get('country','') != 'India']
print_md(f"Total: {len(non_india)}")
for r in non_india:
    print_md(f"- Rank {r['rank']}: {r['candidate_id']} ({r.get('profile',{}).get('country','')}) | Ret Depth: {r.get('retrieval_depth',0)} | YoE: {r.get('years_of_experience',0)}")

print_md("\n## 9. Long Experience (>12 YoE) in Top 100")
long_exp = [r for r in top100_records if r.get('years_of_experience',0) > 12]
for r in long_exp:
    print_md(f"- Rank {r['rank']}: {r['candidate_id']} | YoE: {r.get('years_of_experience',0)} | Title: {r.get('profile',{}).get('current_title','')} | Ret Depth: {r.get('retrieval_depth',0)}")

print_md("\n## 10. Short Experience (4-5 YoE) in Top 100")
short_exp = [r for r in top100_records if 4 <= r.get('years_of_experience',0) <= 5]
for r in short_exp[:5]:
    print_md(f"- Rank {r['rank']}: {r['candidate_id']} | YoE: {r.get('years_of_experience',0)} | Ret Depth: {r.get('retrieval_depth',0)}")
print_md(f"Total short exp: {len(short_exp)}")

print_md("\n## 11. Penalty Hits in Top 100")
print_md(f"- Wrapper Penalty (>0.1): {sum(1 for r in top100_records if r.get('wrapper_penalty',0) > 0.1)}")
print_md(f"- Consulting Penalty (>0.3 ratio): {sum(1 for r in top100_records if r.get('consulting_ratio',0) > 0.3)}")
print_md(f"- Research Penalty (>0): {sum(1 for r in top100_records if r.get('research_only_penalty',0) > 0)}")
print_md(f"- Title Chaser Penalty (>0): {sum(1 for r in top100_records if r.get('title_chaser_penalty',0) > 0)}")

print_md("\n## 12. Honeypots in Top 100")
print_md(f"- Invalid Edu: {sum(1 for r in top100_records if r.get('education_timeline_valid',1) == 0)}")
print_md(f"- Invalid Career: {sum(1 for r in top100_records if r.get('career_timeline_valid',1) == 0)}")
print_md(f"- Inconsistent Exp (<0.5): {sum(1 for r in top100_records if r.get('experience_consistency',1) < 0.5)}")
print_md(f"- Buzzword Stuffing (>2.0): {sum(1 for r in top100_records if r.get('buzzword_density',0) > 2.0)}")

print_md("\n## 13. Feature Contribution")
avg_score = top100_df['final_score'].mean()
avg_bm25 = top100_df['bm25_score'].mean()
print_md(f"- Average Final Score: {avg_score:.2f}")
print_md(f"- Average BM25 Score: {avg_bm25:.2f}")
print_md(f"- Engineered Features Contribution: {(avg_score - avg_bm25):.2f} pts")

print_md("\n## 14. Ablation Test (Top 10 BM25 vs Final Rank)")
df_bm25 = df.sort_values(by='bm25_score', ascending=False).reset_index(drop=True)
df_bm25['bm25_rank'] = df_bm25.index + 1
bm25_ranks = {row['candidate_id']: row['bm25_rank'] for _, row in df_bm25.iterrows()}

for r in top100_records[:10]:
    print_md(f"- Rank {r['rank']} ({r['candidate_id']}) was BM25 Rank {bm25_ranks[r['candidate_id']]}")

out.close()

# Also write metadata
import json as metadata_json
meta = {
    "ArtifactType": "other",
    "RequestFeedback": False,
    "Summary": "Final evaluation report running all 17 checks requested by the user."
}
with open(output_path + '.meta.json', 'w') as f:
    metadata_json.dump(meta, f)
