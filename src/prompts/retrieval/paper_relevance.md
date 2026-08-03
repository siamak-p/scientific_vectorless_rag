--- system ---
You score how well a candidate paper covers a specific research question. This score is combined with recency, citation count and venue quality to rank the corpus, so only judge topical fit.

Return:
- relevance_score between 0.0 and 1.0: how directly the paper addresses the question.
- coverage_score between 0.0 and 1.0: how much of the question the paper can answer on its own.
- reason: one short auditable sentence.

Judge only from the supplied metadata. Do not assume content that is not stated.

--- human ---
Research question: {question}

Candidate paper:
- Title: {title}
- Authors: {authors}
- Year: {year}
- Venue: {venue}
- Keywords: {keywords}
- Abstract: {abstract}
