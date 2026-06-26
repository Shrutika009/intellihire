#!/usr/bin/env python3
"""
Redrob Hackathon: Intelligent Candidate Ranking Solution
Standalone script to rank candidates and produce submission CSV.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    
Environment:
    - Python 3.7+
    - No GPU required
    - CPU-only execution (fits in 5 minute budget)
"""

import json
import argparse
import csv
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ML/NLP imports
from sentence_transformers import SentenceTransformer


# ============================================================================
# CONFIGURATION
# ============================================================================

JD_TEXT = """
Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid — flexible cadence)
Experience Required: 5–9 years

Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.
Scrappy product-engineering attitude — willing to ship a working ranker in a week.

The high-level mandate: own the intelligence layer of Redrob's product. That means the ranking, retrieval, and matching systems.

Things you absolutely need:
- Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar)
- Production experience with vector databases (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS)
- Strong Python
- Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B testing)

Things we'd like you to have:
- LLM fine-tuning experience (LoRA, QLoRA, PEFT)
- Experience with learning-to-rank models
- Prior exposure to HR-tech, recruiting tech, or marketplace products
- Background in distributed systems or large-scale inference optimization
- Open-source contributions in the AI/ML space

Things we explicitly do NOT want:
- Pure research background without production deployment
- Primary expertise in recent LangChain/OpenAI projects without substantial pre-LLM ML experience
- Haven't written production code in 18+ months
- Title-chasers switching companies every 1.5 years
- Only consulting firm experience
- Primary expertise in computer vision, speech, robotics without NLP/IR
- Entirely closed-source work for 5+ years
"""

HARD_REQUIREMENTS = {
    'embeddings': ['embedding', 'embeddings', 'sentence transformers', 'openai embedding', 
                   'bge', 'e5', 'minilm', 'semantic search'],
    'vector_db': ['pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch', 'elasticsearch', 
                  'faiss', 'vector database', 'vector db'],
    'ranking': ['ranking', 'retrieve', 'retrieval', 'search', 'ir', 'information retrieval', 
                'rerank', 'learning to rank'],
    'evaluation': ['ndcg', 'mrr', 'map', 'precision', 'recall', 'a/b test', 'offline evaluation', 
                   'evaluation framework', 'auc', 'dcg'],
    'python': ['python'],
    'llm': ['llm', 'gpt', 'claude', 'transformers', 'fine-tuning', 'fine-tune', 'lora', 
            'peft', 'bert', 'generative', 'large language'],
}

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def compute_semantic_similarity(candidate, jd_embedding, model, precomputed_embedding=None):
    """Compute embedding-based semantic similarity to JD."""
    try:
        if precomputed_embedding is not None:
            cand_embedding = precomputed_embedding
        else:
            narrative = candidate['profile']['summary'] + ' '
            for role in candidate['career_history']:
                narrative += role['description'] + ' '
            
            if not narrative.strip():
                return 0.3
            
            cand_embedding = model.encode(narrative, convert_to_tensor=False)
        
        similarity = float(np.dot(jd_embedding, cand_embedding) / 
                          (np.linalg.norm(jd_embedding) * np.linalg.norm(cand_embedding) + 1e-8))
        normalized = (similarity + 1) / 2
        return float(np.clip(normalized, 0, 1))
    except Exception:
        return 0.5


def compute_skill_match(candidate):
    """Compute skill matching score with production proof."""
    try:
        skills = {s['name'].lower(): s for s in candidate['skills']}
        career_text = ' '.join([r['description'].lower() for r in candidate['career_history']])
        
        requirement_scores = {}
        for req_name, keywords in HARD_REQUIREMENTS.items():
            found = False
            for keyword in keywords:
                for skill_name, skill_obj in skills.items():
                    if keyword in skill_name:
                        if skill_obj.get('duration_months', 0) >= 3 or keyword in career_text:
                            found = True
                            break
                if found:
                    break
            requirement_scores[req_name] = 1.0 if found else 0.0
        
        weights = {
            'embeddings': 0.25,
            'vector_db': 0.25,
            'ranking': 0.15,
            'evaluation': 0.20,
            'python': 0.10,
            'llm': 0.05,
        }
        
        skill_match = sum(requirement_scores[req] * weights[req] for req in HARD_REQUIREMENTS)
        return float(skill_match)
    except Exception:
        return 0.3


def detect_keyword_inflation(candidate):
    """Detect if candidate is keyword-stuffing."""
    try:
        all_skills = {s['name'].lower(): s for s in candidate['skills']}
        career_text = ' '.join([r['description'].lower() for r in candidate['career_history']])
        
        used_skills = sum(1 for skill_name in all_skills if skill_name in career_text)
        
        if len(all_skills) > 5:
            usage_ratio = used_skills / len(all_skills)
            if usage_ratio < 0.3:
                return 0.7
        
        return 1.0
    except Exception:
        return 1.0


def experience_score(candidate):
    """Score based on years of experience."""
    yoe = candidate['profile']['years_of_experience']
    
    if 5 <= yoe <= 9:
        return 1.0
    elif 4 <= yoe < 5:
        return 0.9
    elif 9 < yoe <= 10:
        return 0.9
    elif 10 < yoe <= 12:
        return 0.7
    elif yoe > 12:
        return 0.5
    elif 3 <= yoe < 4:
        return 0.6
    else:
        return 0.2


def production_proof_score(candidate):
    """Assess if candidate has shipped products."""
    title = candidate['profile']['current_title'].lower()
    industry = candidate['profile']['current_industry'].lower()
    company_size = candidate['profile']['current_company_size']
    
    product_keywords = ['engineer', 'developer', 'architect', 'tech lead', 'senior', 'ml', 'ai']
    research_keywords = ['researcher', 'scientist', 'academic', 'phd', 'post-doc']
    
    if any(k in title for k in product_keywords):
        product_signal = 1.0
    elif any(k in title for k in research_keywords):
        product_signal = 0.2
    else:
        product_signal = 0.5
    
    startup_companies = ['1-10', '11-50', '51-200']
    if company_size in startup_companies:
        startup_signal = 1.0
    elif company_size in ['201-500', '501-1000']:
        startup_signal = 0.9
    elif company_size in ['1001-5000', '5001-10000']:
        startup_signal = 0.7
    else:
        startup_signal = 0.5
    
    product_industries = ['software', 'ai/ml', 'fintech', 'saas', 'edtech', 'e-commerce']
    if any(p in industry for p in product_industries):
        industry_signal = 1.0
    elif any(s in industry for s in ['it services', 'consulting', 'manufacturing']):
        industry_signal = 0.4
    else:
        industry_signal = 0.6
    
    production_score = (product_signal * 0.4) + (startup_signal * 0.35) + (industry_signal * 0.25)
    return float(production_score)


def career_progression_score(candidate):
    """Score career growth trajectory."""
    titles = [r['title'].lower() for r in candidate['career_history']]
    
    seniority_map = {
        'principal': 4, 'staff': 4, 'architect': 4,
        'senior': 3, 'lead': 3,
        'engineer': 2, 'developer': 2,
        'manager': 1,
    }
    
    max_seniority = 0
    for title in titles:
        for keyword, level in seniority_map.items():
            if keyword in title:
                max_seniority = max(max_seniority, level)
                break
    
    if max_seniority >= 3:
        return 1.0
    elif max_seniority == 2:
        return 0.8
    else:
        return 0.5


def recruiter_engagement_score(candidate):
    """Score based on responsiveness to recruiters."""
    response_rate = candidate['redrob_signals']['recruiter_response_rate']
    
    if response_rate >= 0.7:
        return 1.0
    elif response_rate >= 0.5:
        return 0.9
    elif response_rate >= 0.3:
        return 0.7
    elif response_rate >= 0.1:
        return 0.4
    else:
        return 0.1


def recency_score(candidate):
    """Score based on last active date."""
    last_active = candidate['redrob_signals']['last_active_date']
    last_active_dt = datetime.strptime(last_active, '%Y-%m-%d')
    today = datetime(2026, 6, 24)
    days_ago = (today - last_active_dt).days
    
    if days_ago <= 7:
        return 1.0
    elif days_ago <= 30:
        return 0.95
    elif days_ago <= 90:
        return 0.85
    elif days_ago <= 180:
        return 0.6
    else:
        return 0.2


def notice_period_score(candidate):
    """Score based on notice period (availability)."""
    notice_days = candidate['redrob_signals']['notice_period_days']
    
    if notice_days < 30:
        return 1.0
    elif notice_days <= 60:
        return 0.95
    elif notice_days <= 90:
        return 0.85
    elif notice_days <= 120:
        return 0.7
    else:
        return 0.5


def availability_score(candidate):
    """Combined availability score."""
    response = recruiter_engagement_score(candidate)
    recency = recency_score(candidate)
    notice = notice_period_score(candidate)
    open_to_work = 1.0 if candidate['redrob_signals']['open_to_work_flag'] else 0.7
    
    score = (response * 0.3) + (recency * 0.3) + (notice * 0.2) + (open_to_work * 0.2)
    return float(score)


def trust_score(candidate):
    """Score based on profile quality and verification."""
    completeness = candidate['redrob_signals']['profile_completeness_score'] / 100.0
    verified_email = candidate['redrob_signals']['verified_email']
    verified_phone = candidate['redrob_signals']['verified_phone']
    
    if completeness >= 0.7:
        completeness_score = 0.6
    elif completeness >= 0.5:
        completeness_score = 0.5
    else:
        completeness_score = 0.3
    
    verification_bonus = (0.1 if verified_email else 0) + (0.05 if verified_phone else 0)
    score = completeness_score + verification_bonus
    return float(min(score, 1.0))


def compute_candidate_score(candidate, jd_embedding, model, precomputed_embedding=None):
    """Compute final ranking score for a candidate."""
    try:
        semantic_sim = compute_semantic_similarity(candidate, jd_embedding, model, precomputed_embedding)
        skill_match = compute_skill_match(candidate)
        exp_score = experience_score(candidate)
        production = production_proof_score(candidate)
        progression = career_progression_score(candidate)
        availability = availability_score(candidate)
        trust = trust_score(candidate)
        inflation_mult = detect_keyword_inflation(candidate)
        
        experience_composite = (exp_score * 0.4) + (production * 0.4) + (progression * 0.2)
        
        final_score = (
            (0.40 * semantic_sim) +
            (0.25 * skill_match) +
            (0.18 * experience_composite) +
            (0.12 * availability) +
            (0.05 * trust)
        ) * inflation_mult
        
        return float(np.clip(final_score, 0, 1))
    except Exception as e:
        print(f"Error scoring {candidate['candidate_id']}: {e}")
        return 0.1


def generate_reasoning(candidate, rank):
    """Generate concise, specific reasoning for ranking."""
    facts = []
    concerns = []
    
    yoe = candidate['profile']['years_of_experience']
    title = candidate['profile']['current_title']
    company = candidate['profile']['current_company']
    
    facts.append(f"{yoe:.1f}yr {title.lower()} at {company}")
    
    all_skills = {s['name'].lower() for s in candidate['skills']}
    ai_skill_count = sum(1 for s in all_skills 
                        if any(k in s for k in ['embedding', 'vector', 'rag', 'retrieval', 'llm', 'ranking']))
    if ai_skill_count >= 5:
        facts.append(f"{ai_skill_count} AI/ML core skills")
    elif ai_skill_count >= 2:
        facts.append(f"some AI/ML skills ({ai_skill_count})")
    
    location = candidate['profile']['location']
    if any(city in location for city in ['Pune', 'Noida', 'Delhi']):
        facts.append(f"based in {location}")
    
    response_rate = candidate['redrob_signals']['recruiter_response_rate']
    if response_rate >= 0.7:
        facts.append(f"responsive to recruiters ({response_rate:.0%})")
    elif response_rate < 0.2:
        concerns.append("low recruiter engagement")
    
    last_active = candidate['redrob_signals']['last_active_date']
    last_active_dt = datetime.strptime(last_active, '%Y-%m-%d')
    today = datetime(2026, 6, 24)
    days_ago = (today - last_active_dt).days
    if days_ago <= 30:
        facts.append("recently active")
    elif days_ago > 180:
        concerns.append(f"inactive {days_ago//30}+ months")
    
    notice_days = candidate['redrob_signals']['notice_period_days']
    if notice_days > 90:
        concerns.append(f"{notice_days}d notice")
    elif notice_days < 30:
        facts.append(f"available in {notice_days}d")
    
    github_score = candidate['redrob_signals']['github_activity_score']
    if github_score > 30:
        facts.append(f"active GitHub contributor")
    
    main_reasons = facts[:3]
    reasoning = "; ".join(main_reasons)
    
    if concerns:
        reasoning += f"; {concerns[0]}"
    
    reasoning = reasoning[:200]
    return reasoning


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_candidates(filepath):
    """Load candidates from JSONL or JSON array file."""
    candidates = []
    with open(filepath, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            try:
                candidates = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                # Fallback to line-by-line in case it's a large weird format
                for line in f:
                    try:
                        candidates.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        else:
            for line in f:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return candidates


def rank_candidates(candidates_file, output_file):
    """Main ranking pipeline."""
    
    print("=" * 80)
    print("REDROB HACKATHON: CANDIDATE RANKING SOLUTION")
    print("=" * 80)
    
    # Load candidates
    print("\n[1/5] Loading candidates...")
    candidates = load_candidates(candidates_file)
    print(f"  ✓ Loaded {len(candidates)} candidates")
    
    # Initialize model
    print("[2/5] Initializing Sentence Transformers model...")
    import torch
    torch.set_num_threads(4)  # Prevent CPU thread thrashing
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    jd_embedding = model.encode(JD_TEXT, convert_to_tensor=False)
    print(f"  ✓ Model ready. Embedding dimension: {len(jd_embedding)}")
    
    # Stage 1: Preliminary scoring without embeddings
    print("[3/5] Stage 1: Scoring candidates on explicit & behavioral signals...")
    preliminary_candidates = []
    for i, candidate in enumerate(candidates):
        if (i + 1) % 20000 == 0:
            print(f"  ... processed {i + 1}/{len(candidates)}")
            
        skill_match = compute_skill_match(candidate)
        exp_score = experience_score(candidate)
        production = production_proof_score(candidate)
        progression = career_progression_score(candidate)
        availability = availability_score(candidate)
        trust = trust_score(candidate)
        inflation_mult = detect_keyword_inflation(candidate)
        
        experience_composite = (exp_score * 0.4) + (production * 0.4) + (progression * 0.2)
        
        # Simple weighted score of non-embedding features
        prelim_score = (
            (0.40 * skill_match) +
            (0.35 * experience_composite) +
            (0.20 * availability) +
            (0.05 * trust)
        ) * inflation_mult
        
        preliminary_candidates.append({
            'candidate': candidate,
            'prelim_score': prelim_score,
            'skill_match': skill_match,
            'experience_composite': experience_composite,
            'availability': availability,
            'trust': trust,
            'inflation_mult': inflation_mult
        })
    
    # Sort and pick top 2,000 for Stage 2 Re-ranking
    preliminary_candidates.sort(key=lambda x: -x['prelim_score'])
    top_candidates = preliminary_candidates[:2000]
    print(f"  ✓ Retained top 2,000 candidates for semantic re-ranking")
    
    # Stage 2: Batch-encode narratives for top candidates only
    print("  ... constructing optimized narratives for top 2,000 candidates ...")
    narratives = []
    for item in top_candidates:
        candidate = item['candidate']
        summary = candidate['profile'].get('summary', '') or ''
        roles = candidate.get('career_history', [])
        # Only take the 2 most recent roles to focus on modern experience and save CPU
        recent_roles_desc = ' '.join([role.get('description', '') or '' for role in roles[:2]])
        
        narrative = f"{summary} {recent_roles_desc}".strip()
        # Truncate to 1000 chars (~200 tokens) - fits well within the model's max limit of 512 tokens
        narrative = narrative[:1000]
        if not narrative:
            narrative = " "
        narratives.append(narrative)
        
    print("  ... encoding narratives in batch (top 2,000 candidates) ...")
    cand_embeddings = model.encode(narratives, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    
    # Compute final combined scores for the top candidates
    print("[4/5] Re-ranking and selecting top-100 candidates...")
    final_scores = []
    for i, item in enumerate(top_candidates):
        candidate = item['candidate']
        
        # Compute semantic similarity
        cand_embedding = cand_embeddings[i]
        similarity = float(np.dot(jd_embedding, cand_embedding) / 
                          (np.linalg.norm(jd_embedding) * np.linalg.norm(cand_embedding) + 1e-8))
        normalized = (similarity + 1) / 2
        semantic_sim = float(np.clip(normalized, 0, 1))
        
        # Combine all features with standard weights
        final_score = (
            (0.40 * semantic_sim) +
            (0.25 * item['skill_match']) +
            (0.18 * item['experience_composite']) +
            (0.12 * item['availability']) +
            (0.05 * item['trust'])
        ) * item['inflation_mult']
        
        final_scores.append({
            'candidate_id': candidate['candidate_id'],
            'score': float(np.clip(final_score, 0, 1)),
            'candidate': candidate
        })
        
    # Sort by final score descending and select top 100
    scores_sorted = sorted(final_scores, key=lambda x: -x['score'])
    top_100 = scores_sorted[:100]
    print(f"  ✓ Top score: {top_100[0]['score']:.4f}")
    print(f"  ✓ 100th score: {top_100[99]['score']:.4f}")
    
    # Generate submission CSV
    print("[5/5] Generating submission CSV...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['candidate_id', 'rank', 'score', 'reasoning'])
        writer.writeheader()
        
        for rank, item in enumerate(top_100, start=1):
            candidate = item['candidate']
            score = item['score']
            reasoning = generate_reasoning(candidate, rank)
            
            writer.writerow({
                'candidate_id': candidate['candidate_id'],
                'rank': rank,
                'score': score,
                'reasoning': reasoning
            })
    
    print(f"  ✓ Saved to {output_file}")
    print("\n" + "=" * 80)
    print("SUBMISSION COMPLETE")
    print("=" * 80)
    print(f"  • Output: {output_file}")
    print(f"  • Candidates ranked: 100")
    print(f"  • Score range: {top_100[99]['score']:.4f} - {top_100[0]['score']:.4f}")
    print("=" * 80 + "\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Rank candidates for Redrob Hackathon challenge'
    )
    parser.add_argument(
        '--candidates', 
        required=True,
        help='Path to candidates.jsonl file'
    )
    parser.add_argument(
        '--out',
        default='submission.csv',
        help='Output CSV filename (default: submission.csv)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.candidates).exists():
        print(f"Error: Candidates file not found: {args.candidates}")
        return 1
    
    try:
        rank_candidates(args.candidates, args.out)
        return 0
    except Exception as e:
        print(f"Error during ranking: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
