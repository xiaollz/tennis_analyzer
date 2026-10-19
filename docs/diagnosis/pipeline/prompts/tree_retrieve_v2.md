You are given the transcript of a tennis coaching video, and a library of problem nodes and cause nodes that were extracted from OTHER videos. The video has never been used to build the library. A cause node may be followed by "also worded:" lines, which are other statements already known to describe the same cause.

Do two independent tasks.

TASK A (problems): which library problem nodes describe the problem this video sets out to solve? The list may be empty if none does.

TASK B (causes): for each root cause the coach explains in this video, find the library cause nodes that describe the SAME mechanism. Answer TASK B on its own: cause nodes are not tied to problem nodes, and an empty or doubtful TASK A must not stop you from answering TASK B. Judge every cause node only on its own wording.

Rules:
- Choose only ids that appear in the library. Never invent an id.
- A node matches only if it describes the SAME mechanism as what the coach says in this video. The same stroke or the same body part is not enough.
- Confidence: "high" only when the coach explicitly states the same mechanism as the node; "medium" for a close match; "low" for a weak match.
- Rank each list best match first. At most 3 problems and at most 5 causes. Return a cause only if it describes a mechanism the coach states; if no cause node does, return an empty causes list.
- If the video is not coaching a technique problem (for example match tactics, equipment, a clip with no teaching), set "video_is_teaching" to false.

Return JSON only:
{"video_is_teaching": true, "problems": [{"id": "N-P_...", "confidence": "high"}], "causes": [{"id": "N-C_...", "confidence": "medium"}]}

LIBRARY (problem nodes, then cause nodes; format: id | text)
{TREE}

TRANSCRIPT (format: [seconds] text)
{TRANSCRIPT}
