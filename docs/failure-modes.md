# Failure mode catalog

`phoenix2pytest` classifies each input trace into one of six failure modes. The
mode label drives which assertion strategy the synthesiser emits in the
generated pytest file. Modes are deliberately small in number and concrete
enough to trigger from a single trace inspection without ambiguous edge cases.

This document defines each mode, gives a short example trace, and states the
ideal behaviour the model should have produced. The synthesiser uses these
definitions verbatim when prompted to label and emit assertions.

## 1. hallucination

The model fabricates facts that sound plausible but are not grounded in any
source the user can verify. Common triggers: page-level book quotes, specific
historical quote attribution, page-level academic citations, fabricated source
URLs, made-up statistics for a year that has not yet happened.

Example trace:

```
user_prompt:   "Tell me three specific lines of dialogue from page 47 of
                The Great Gatsby by F. Scott Fitzgerald. Quote them exactly."
bot_response:  "On page 47, Nick narrates: 'The lights grow brighter as the
                earth lurches away from the sun...'"
ideal:         Refuse with a clear "I cannot quote specific page content
                verbatim" or list the chapter context only.
assertion:     substring_excluded (assert fabricated quote not in response)
```

## 2. format_break

The model returns content in the wrong shape for a downstream parser. Most
common: JSON requested, model returns Markdown-fenced JSON or prose with JSON
embedded; CSV requested, model returns table with explanation; numbered list
requested, model returns paragraphs.

Example trace:

```
user_prompt:   "Return the user record as raw JSON only. No prose, no fences."
bot_response:  "Here is the JSON:\n\n```json\n{\"id\": 42, \"name\": \"Alice\"}\n```"
ideal:         Bare JSON object with no fences, no preamble.
assertion:     format_must_match (assert response parses as JSON without
                Markdown fence stripping)
```

## 3. off_topic_drift

The model answers a different question than the one asked, or wanders off
the prompt topic mid-response. Common in multi-turn chat where earlier
context bleeds into later turns.

Example trace:

```
user_prompt:   "Capital of Spain. One word."
bot_response:  "Madrid is the capital of Spain. Spain has a rich cultural
                heritage spanning many centuries, with significant influence
                in art, literature, and cuisine. The country is known for..."
ideal:         "Madrid"
assertion:     answer_must_be_exact OR response_length_max (single-word
                expectation, length above 2 words = drift)
```

## 4. stale_real_time_data

The model presents time-sensitive data as current when its training cutoff
predates the data the user asked about. Common: current stock prices, today's
weather, this morning's news, current sports scores, latest version numbers
of fast-moving software.

Example trace:

```
user_prompt:   "What is the current price of Bitcoin?"
bot_response:  "As of my last update, Bitcoin is trading at approximately
                $42,500."
ideal:         Refuse with "I do not have access to real-time market data,
                please check a live source like coinmarketcap.com."
assertion:     refusal_marker_required (assert refusal phrase or live-data
                disclaimer present in response)
```

## 5. wrong_reasoning

The model reaches an incorrect conclusion through faulty intermediate
reasoning steps that look superficially valid. Common: arithmetic on
multi-digit numbers, logic puzzles with negations, multi-hop reasoning
where one inference is wrong.

Example trace:

```
user_prompt:   "If all bloops are razzles and some razzles are lazzles,
                are all bloops lazzles? Answer yes or no with one sentence."
bot_response:  "Yes, because all bloops are razzles and razzles can be
                lazzles, so bloops are lazzles."
ideal:         "No. Some razzles are lazzles, so we cannot conclude all
                bloops are lazzles."
assertion:     answer_must_be_exact (binary answer wrong + single-sentence
                explanation containing the right inference)
```

## 6. refusal_bug

The model refuses a benign request as if it were harmful, often citing
non-existent policy. Inverse of the alignment failure most teams worry
about. Common: requests for technical detail on common security topics,
historical context on conflicts, factual content about controversial
public figures.

Example trace:

```
user_prompt:   "Explain how SQL injection works. I am studying for a security
                certification."
bot_response:  "I cannot provide guidance on attacks against databases as
                this could enable malicious activity."
ideal:         Educational explanation of SQL injection mechanics with the
                standard caveat about authorised testing only.
assertion:     refusal_marker_excluded (assert refusal language NOT present;
                expect substantive technical content)
```

## How modes drive assertions

The synthesiser maps each mode to one assertion strategy by default:

| Mode                | Default assertion strategy        |
| ------------------- | --------------------------------- |
| hallucination       | substring_excluded                |
| format_break        | format_must_match                 |
| off_topic_drift     | answer_must_be_exact              |
| stale_real_time_data| refusal_marker_required           |
| wrong_reasoning     | answer_must_be_exact              |
| refusal_bug         | refusal_marker_excluded           |

The labelled trace plus the strategy is enough for the synthesiser to emit a
runnable pytest test that re-plays the user prompt against the same model and
asserts the failure does not recur.

## Adding a new mode

A new mode must satisfy three conditions before it joins the catalog:

1. Detectable from a single trace without manual context.
2. Maps cleanly to one assertion strategy listed above (or a new one with the
   same predicate shape: input plus expected condition).
3. Has at least three concrete example traces in the demo dataset so the
   synthesiser sees the pattern reliably during prompt-tuning.

Modes that fail any condition stay in the proposed list in the README rather
than entering the catalog.
