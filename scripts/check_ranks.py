import csv, json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'c:\Users\admin\Desktop\redrobrproject\final_rankings.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

lines = open(r'c:\Users\admin\Desktop\redrobrproject\data\candidates_featured.jsonl', 'r', encoding='utf-8').readlines()
records = {json.loads(l)['candidate_id']: json.loads(l) for l in lines}

print('=== Top 10 Candidates ===')
for r in rows[:10]:
    cid = r['candidate_id']
    country = records[cid].get('profile', {}).get('country', '?')
    print(f"Rank {r['rank']:<2}: {cid} | Score: {r['score']} | Country: {country}")

non_india = []
for r in rows:
    cid = r['candidate_id']
    country = records[cid].get('profile', {}).get('country', '?')
    if country != 'India':
        non_india.append((r['rank'], cid, country))

print(f'\n=== Non-India Candidates in Top 100: {len(non_india)} ===')
for rank, cid, country in non_india[:10]:
    print(f"Rank {rank:<3}: {cid} ({country})")
if len(non_india) > 10:
    print('...')
