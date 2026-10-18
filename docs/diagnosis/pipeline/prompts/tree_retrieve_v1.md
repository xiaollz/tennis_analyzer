You are given the transcript of a tennis coaching video, and a library of problem nodes and cause nodes that were extracted from OTHER videos. The video above has never been used to build the library.

Task: decide which library nodes describe (a) the problem this video sets out to solve and (b) the root causes the coach explains for it.

Rules:
- Choose only ids that appear in the library. Never invent an id.
- A node matches only if it describes the SAME mechanism as what the coach says in this video. The same stroke or the same body part is not enough.
- If the library has no node for the video's problem or causes, return empty lists and set "none_match" to true. If the video is not coaching a technique problem (for example match tactics, equipment, a clip with no teaching), set "video_is_teaching" to false.
- Confidence: "high" only when the coach explicitly states the same mechanism as the node; "medium" for a close match; "low" for a weak match.
- List at most 3 problems and at most 5 causes, best match first.

Return JSON only:
{"video_is_teaching": true, "none_match": false, "problems": [{"id": "N-P_...", "confidence": "high"}], "causes": [{"id": "N-C_...", "confidence": "medium"}]}

LIBRARY (problem nodes, then cause nodes; format: id | text)
{TREE}

TRANSCRIPT (format: [seconds] text)
{TRANSCRIPT}
