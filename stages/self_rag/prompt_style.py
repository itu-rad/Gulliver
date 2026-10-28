"""Short-answer style instruction shared by every answer-producing Self-RAG stage.

Defined once so the monolithic and decomposed variants request the same answer
format, which EM/F1 scoring assumes. Changing it changes quality results.
"""

SHORT_ANSWER_STYLE = (
    "Answer format (STRICT): reply with only the shortest exact answer — a "
    "name, entity, phrase, number, or 'yes'/'no' — copied from the documents "
    "where possible. Do NOT write a full sentence. No explanations, no "
    "restating the question, no punctuation beyond the answer itself. "
    "Examples: 'yes'; 'Abraham Lincoln'; '1862'; 'the Legal Tender Act'."
)
