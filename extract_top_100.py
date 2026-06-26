import csv
import json

# Load the top 100 candidate IDs and ranks/scores from submission.csv
top_100_meta = {}
with open('submission.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        top_100_meta[row['candidate_id']] = {
            'rank': int(row['rank']),
            'score': float(row['score']),
            'reasoning': row['reasoning']
        }

# Find these candidates in candidates.jsonl
top_100_profiles = []
with open('candidates.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            cand = json.loads(line)
            cid = cand['candidate_id']
            if cid in top_100_meta:
                # Merge metadata
                cand['rank'] = top_100_meta[cid]['rank']
                cand['score'] = top_100_meta[cid]['score']
                cand['reasoning'] = top_100_meta[cid]['reasoning']
                top_100_profiles.append(cand)
        except json.JSONDecodeError:
            continue

# Sort by rank
top_100_profiles.sort(key=lambda x: x['rank'])

with open('profiles.js', 'w', encoding='utf-8') as f:
    f.write("const candidateProfiles = ")
    json.dump(top_100_profiles, f, indent=2)
    f.write(";")

print(f"Extracted {len(top_100_profiles)} profiles to profiles.js")
