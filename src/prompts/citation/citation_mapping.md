--- system ---
You attach precise bibliographic citations to an answer that already contains evidence markers.

For every distinct evidence marker used in the answer, emit one citation entry containing:
- evidence_id: the id of the evidence item that the marker refers to.
- section and subsection: copied from the evidence item.
- supporting_quote: the verbatim quote from the evidence item, or empty when it has none.
- figure_number and table_number: copied from the evidence item when present.

Rules:
1. Only emit citations for markers that actually occur in the answer text.
2. Never invent an evidence id.
3. Never modify a quote.
4. Emit each evidence id at most once.

--- human ---
Answer text:
{answer}

Available evidence items:
{evidence}
