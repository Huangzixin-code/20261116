---
name: multi_hop_identity
description: Resolve a person or entity from several clues by searching clues separately and verifying them against documents.
---
# Multi-hop identity resolution
1. Extract 3–5 clue groups: identity, dates/education, discovery/work, award, and service or location.
2. Search the most discriminative relation first with a short query; do not place the whole question into one broad query. If useful, add up to two independent clue queries in the same search action.
3. Open the strongest candidate documents and keep a ledger of which clue each source supports.
4. Translate broad scientific descriptions into genuine technical synonyms and plausible categories. Do not conflate related terms with different meanings (for example, cytokinin and cytokine). Treat expansions as query alternatives, not facts.
5. Spend each remaining search on one unresolved clue. When a candidate is known, include the candidate name and a distinctive verified clue in a verification query. Never invent dates, names, awards, countries, or events to make a query more specific.
6. Do not give up while a specific clue remains searchable. Verify a candidate against at least two independent clues before answering.
7. Return only the canonical requested answer. Use an insufficient-evidence answer only after the search budget and promising documents have been checked.
