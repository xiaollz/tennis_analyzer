import re
import json
import subprocess
import os
from pathlib import Path

# Read the markdown file
with open('/Users/qsy/Desktop/tennis/learn_ytb/网球学习指南_v2_综合版.md', 'r') as f:
    content = f.read()

# Extract all YouTube video IDs
patterns = [
    r'youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
    r'youtu\.be/([a-zA-Z0-9_-]{11})',
]

video_ids = set()
video_info = []  # List of (section, title, video_id)

# Parse by sections
sections = content.split('\n## ')
current_section = "Introduction"

for section in sections:
    lines = section.split('\n')
    if lines[0].startswith('#'):
        continue
    section_title = lines[0].split('\n')[0].strip()
    if section_title:
        current_section = section_title
    
    for line in lines:
        # Find video links with titles
        match = re.search(r'\[([^\]]+)\]\((https?://[^\)]+youtube[^\)]+)\)', line)
        if match:
            title = match.group(1)
            url = match.group(2)
            for pattern in patterns:
                id_match = re.search(pattern, url)
                if id_match:
                    vid = id_match.group(1)
                    if vid not in video_ids:
                        video_ids.add(vid)
                        video_info.append({
                            'section': current_section,
                            'title': title,
                            'video_id': vid,
                            'url': url
                        })
                    break

print(f"Found {len(video_info)} unique videos")

# Group by section
sections_count = {}
for v in video_info:
    sec = v['section']
    sections_count[sec] = sections_count.get(sec, 0) + 1

print("\nVideos per section:")
for sec, count in sections_count.items():
    print(f"  {sec}: {count}")

# Save to JSON for processing
output_path = '/Users/qsy/Desktop/tennis/video_list.json'
with open(output_path, 'w') as f:
    json.dump(video_info, f, ensure_ascii=False, indent=2)

print(f"\nSaved video list to {output_path}")
