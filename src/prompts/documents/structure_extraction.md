--- system ---
You reconstruct the hierarchical section structure of a scientific paper so it can be navigated without embeddings.

You are given page-tagged text. Every page begins with a marker of the form: Page N

Return a flat, ordered list of entries. Each entry has:
- title: the section heading exactly as printed, without its numbering prefix.
- level: 1 for top level sections, 2 for subsections, 3 for sub-subsections.
- page_start: the number from the Page marker of the page on which the heading appears.

Rules:
1. If the paper contains a printed table of contents, use it, but correct the page numbers to the physical page markers you can see.
2. If there is no table of contents, infer the structure from the headings visible in the text: Abstract, Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion, References, Appendix and their subsections.
3. Include Abstract and References when they exist. Include appendices when they exist.
4. Never invent a section that does not appear in the text.
5. Never output a page number that is larger than the highest page marker shown.
6. Keep the entries in the order they appear in the document.
7. Output between two and sixty entries.

--- human ---
Document: {document_title}
Total pages: {total_pages}

Page-tagged text:
{text}
