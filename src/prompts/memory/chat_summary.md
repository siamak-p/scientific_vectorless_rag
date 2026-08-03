--- system ---
Summarise a conversation at a topic level only.

This summary is shared across conversations, so it must stay general. It exists so the assistant can say what subject areas the user has worked on before, without ever recalling the specifics of another conversation.

Return:
- topics: three to eight short subject labels, for example: transformer architectures, medical image segmentation, dataset bias.
- summary: one or two sentences describing the subject area at a high level.

Rules:
1. No numbers, no results, no dataset names, no paper titles, no author names, no quotes.
2. No details that would let someone reconstruct what was actually said.
3. If the conversation has no substantive subject, return an empty topic list and an empty summary.

--- human ---
Conversation title: {title}

Messages:
{messages}
