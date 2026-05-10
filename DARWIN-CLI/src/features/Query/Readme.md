

seperate query from metadata
if large text, split from query and tag largetext 
users can tag largetexts by themself

first check files refrenced in query
using pinnedfiles (if the user refrences fils in the query...we can fix that with @ for file recongition and extractiong with regex)
files refrenceed without @ will not be refrenced
add refrencd files to pinned files if they are not already there and filter the duplicates replace the file and replace it with just the filename in the query instead of the @ refrence

loop over files and check if they have been proceesed
if not processed...push back on the user 
"filex is not processed yet would you like to index it now?"

after we do a good job of seprating text by linespaces

we check if what was pasteed is a query or not like doc refrences, code, orr just large text
this is what seperates the query from the metadata


todo: maybe we check if the user request is vulgar or offensive before we send it to the llm

now we have ::
- query
- metadata(pasted texts, links, pinned files, isfollowup:: like is this a followup to a previous query or not)

we pass this to the small llm to get back an oranizzed query


@dataclass
class OrganizedQuery:
    clean_query: str
    questionType : list[QuestionType]
    suggestedRepresentation: list[ResponsePresentation]
    pasted_texts: list[PastedText]
    extracted_urls: list[URLRef]
    pinned_file_ids: list[str]
    is_followup: bool
    needs_clarification: bool
    followup: list[FollowUpBlock] | None



this oranizzed query will be used to search the index
the organizedquery will contain
- query
- questiontype :   "fact" |  "instruction" | "reasoning" |   "comparison" | "opinion" | "hypothetical" | "clarification" |  "multi_part" | "yes_no" | "statement"
suggestedRepresentation : a list of suggsted components to reply with
- pasted texts
- extracted urls
- pinned files










You are a Query Organizer for a retrieval system.

Your job is NOT to answer the user.

Your job is ONLY to transform a parsed user request into a structured OrganizedQuery object.

You MUST follow these rules strictly:

---

RULES:

1. NEVER answer the question.
2. NEVER include explanations.
3. NEVER include raw blob contents.
4. NEVER invent fields not defined in the schema.
5. ALWAYS output valid JSON only.
6. NEVER return markdown, text, or commentary.
7. If uncertain, use empty arrays or false values.
8. Do not merge retrieval_query with clean_query.
9. retrieval_query must be independent search phrases, NOT rewritten user queries.
10. Do not duplicate information across fields.

---



---

OUTPUT MUST MATCH SCHEMA EXACTLY.
No extra keys allowed.
```

---

# 🧩 JSON SCHEMA CONTRACT

This is your **hard output contract**.

```json
{
  "clean_query": "string",

  "question_type": ["fact | instruction | reasoning | comparison | opinion | hypothetical | clarification | multi_part | yes_no | statement"],

  "retrieval_query": "string",

  "suggested_representation": list[
     "plain_text",
    "markdown",
    "accordion",
    "comparison_table",
    "stats_grid",
    "summary_card",
    "timeline",
    "entity_map",
    "flowchart",
    "bar_chart",
    "stepper",
    "graph",


    ],


}
```

---

# 🧠 USER PROMPT TEMPLATE

You pass this to the model:

```text
You are given a structured parsed input.

Transform it into an OrganizedQuery.

---

INPUT:

query:
{clean_query}



---

TASK:
Produce ONLY the JSON output that matches the schema exactly.
```

---

# 🔥 CRITICAL DESIGN GUARANTEES

## 1. No hallucinated structure

Because:

* schema is fixed
* system prompt forbids extra keys
* no reasoning allowed

---

## 2. Retrieval safety

### retrieval_query MUST be:

✔ short
✔ independent
✔ search-engine friendly
❌ not rewritten user query
❌ not blob-dependent

---

## 3. Blob safety

The model sees:

```text
[BLOB:123,log]
```

NOT content.

So it CANNOT:

* leak sensitive data
* overfit retrieval terms
* hallucinate based on content

---

## 4. Field isolation (very important)

Each field has ONE job:

| Field                    | Purpose           |
| ------------------------ | ----------------- |
| clean_query              | normalized intent |
| retrieval_query          | search strategy   |
| blob_ids                 | references only   |
| file_ids                 | scope             |
| suggested_representation | UI planning       |

No overlap allowed.

---

# 🧠 Retrieval Query Rules (important addition)

Add this inside system prompt if you want stronger control:

```text
retrieval_query rules:

- Each entry must be a standalone search phrase
- Must be <= 12 words
- Must NOT include file names or blob IDs
- Must NOT repeat user query verbatim
- Must expand intent into search terms
- Prefer technical / semantic phrasing
```

---

# 🧪 Example behavior

## Input:

```text
why is my api slow when uploading files @file:logs.txt
```

## Output:

```json
{
  "clean_query": "why is my api slow when uploading files",

  "question_type": ["reasoning", "instruction"],

  "retrieval_query": "api file upload performance bottleneck backend request latency api upload debugging logs analysis",

  "suggested_representation": ["explanation", "diagnosis"],

}
```

---
