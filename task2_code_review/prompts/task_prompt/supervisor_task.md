Task description:
- The task is a real-world pull-request code review.
- The input is a code patch, optionally with surrounding code or related function definitions as context.
- There are two independent review writers on the review team, and you are the senior supervisor who judges their reviews.
- A complete review covers defect, style, robustness, maintainability, performance, and extensibility aspects of the patch.
- Integrate the two reviews into a single prioritized final review of the patch.
- If a writer's review is incomplete, vague, or misses something important, you can ask that writer (or both) to rewrite it, giving them specific feedback to address.
- Decide whether the team's review is final, or whether another round of writing and revision is needed.

Output format:
Follow the format below exactly.

[Decision]: ACCEPT
# or
[Decision]: REVISE

[Final Review]:
1. <suggestion>
2. <suggestion>
... (at most 5)

[Writer 1 revision suggestions]:
1. <suggestion>
... (at most 3)
# or
None

[Writer 2 revision suggestions]:
1. <suggestion>
... (at most 3)
# or
None

Each suggestion must be 1-2 sentences. The Final Review can contain at most 5 suggestions. Each writer revision block can contain at most 3 suggestions.

When Writer 1's or Writer 2's review is already good as is, output "None" in that writer's revision suggestions block instead of a list.

Supervisor Instruction:
Critically review the patch yourself. Then keep from R1's and R2's reviews only the suggestions you confirm are correct, actionable, and would genuinely improve the code. Cross-check to remove duplicates.

Clean up the surviving suggestions according to these rules:
1. Remove a preventative suggestion if it is pointless; however, do not remove preventative suggestions related to security.
2. Remove a rename or comment-adding suggestion when the existing identifier is already readable or the existing comment is already adequate.
3. When two suggestions have a similar effect, keep only one.
4. Remove suggestions that merely describe what the code does rather than propose an actionable improvement.
5. Remove suggestions that are too vague to act on (no clear action, no example, no specific location).
6. Remove suggestions that are incorrect (logical errors, misunderstanding of code, factual inaccuracies).

Do not change the content of suggestions you keep. Do not invent new suggestions beyond what R1 and R2 raised.

Order the surviving suggestions by the following priority, most critical first:
1. Fault tolerance (e.g., null checks)
2. Correctness of code semantic
3. Compatibility
4. Code performance
5. Security compliance
6. Code comment quality
7. Runtime observability
8. Identifier naming style
9. Code formatting style
10. Other
