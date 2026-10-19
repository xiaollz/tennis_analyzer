You are checking whether items from one list describe the same mechanism as items from another list. You see only short statements, no videos.

LIST A (gold items, from one coaching video):
{GOLD}

LIST B (library nodes; format: id | text):
{TREE}

For every item in list A, return the ids from list B whose statement describes the SAME mechanism (same body action or same underlying idea, even if worded differently). Same stroke or same body part alone is NOT enough. Return at most 3 ids per item, best first; an empty list is the right answer when nothing matches. Problem items (ids starting "G-P") may only match ids starting "N-P_"; cause items (ids starting "G-C") may only match ids starting "N-C_".

Return JSON only: {"matches": {"G-P": ["N-P_..."], "G-C1": ["N-C_..."]}}
