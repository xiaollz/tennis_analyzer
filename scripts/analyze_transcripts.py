import json
import os
from pathlib import Path
from collections import defaultdict

TRANSCRIPTS_DIR = Path("/Users/qsy/.gemini/transcripts")
VIDEO_LIST = Path("/Users/qsy/Desktop/tennis/video_list.json")

# Load video metadata
with open(VIDEO_LIST, 'r') as f:
    videos = json.load(f)

# Create video ID to metadata mapping
video_map = {v['video_id']: v for v in videos}

# Organize transcripts by section
sections = defaultdict(list)

for transcript_file in TRANSCRIPTS_DIR.glob("*.json"):
    vid = transcript_file.stem
    if vid not in video_map:
        continue
    
    meta = video_map[vid]
    section = meta['section']
    title = meta['title']
    
    with open(transcript_file, 'r') as f:
        data = json.load(f)
    
    if 'error' in data or 'lines' not in data:
        continue
    
    # Extract full transcript text
    transcript_text = " ".join([l.get("text", "") for l in data.get("lines", [])])
    
    sections[section].append({
        'title': title,
        'video_id': vid,
        'transcript': transcript_text,
        'word_count': len(transcript_text.split())
    })

# Print summary
print("=" * 60)
print("TRANSCRIPT ANALYSIS SUMMARY")
print("=" * 60)

for section, vids in sections.items():
    total_words = sum(v['word_count'] for v in vids)
    print(f"\n📚 {section}")
    print(f"   Videos: {len(vids)} | Total Words: {total_words:,}")
    for v in vids[:3]:  # Show first 3
        print(f"   - {v['title'][:50]}... ({v['word_count']:,} words)")
    if len(vids) > 3:
        print(f"   ... and {len(vids)-3} more")

# Extract key concepts from each section
print("\n" + "=" * 60)
print("KEY CONCEPTS EXTRACTION")
print("=" * 60)

# Define key terms to search for in transcripts
key_terms = {
    'forehand': ['unit turn', 'contact point', 'hip', 'shoulder', 'wrist', 'lag', 'topspin', 'follow through', 'swing path', 'grip'],
    'backhand': ['one handed', 'grip', 'shoulder', 'extension', 'slice', 'topspin', 'footwork', 'preparation'],
    'serve': ['toss', 'pronation', 'trophy', 'power', 'kick', 'slice', 'flat', 'rhythm', 'leg drive'],
    'volley': ['contact', 'punch', 'continental', 'split step', 'ready position', 'soft hands'],
    'footwork': ['split step', 'recovery', 'open stance', 'neutral stance', 'weight transfer'],
    'tactics': ['cross court', 'down the line', 'approach', 'defensive', 'offensive']
}

# Count term occurrences in transcripts
term_counts = defaultdict(lambda: defaultdict(int))

for section, vids in sections.items():
    all_text = " ".join([v['transcript'].lower() for v in vids])
    for category, terms in key_terms.items():
        for term in terms:
            count = all_text.count(term.lower())
            if count > 0:
                term_counts[section][term] = count

for section, counts in term_counts.items():
    if counts:
        print(f"\n📊 {section}")
        sorted_terms = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for term, count in sorted_terms:
            print(f"   '{term}': {count} mentions")

# Save detailed analysis
output = {
    'sections': {},
    'total_transcripts': sum(len(v) for v in sections.values()),
    'total_words': sum(sum(x['word_count'] for x in v) for v in sections.values())
}

for section, vids in sections.items():
    output['sections'][section] = {
        'video_count': len(vids),
        'total_words': sum(v['word_count'] for v in vids),
        'videos': [{'title': v['title'], 'video_id': v['video_id'], 'word_count': v['word_count']} for v in vids]
    }

with open('/Users/qsy/Desktop/tennis/knowledge_analysis.json', 'w') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ Analysis saved to knowledge_analysis.json")
print(f"📊 Total: {output['total_transcripts']} transcripts, {output['total_words']:,} words")
