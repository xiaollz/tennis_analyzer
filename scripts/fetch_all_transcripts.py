import json
import subprocess
import os
import time
from pathlib import Path

# MCP server path
MCP_PATH = "/Users/qsy/.gemini/youtube-summarizer-mcp"
OUTPUT_DIR = Path("/Users/qsy/.gemini/transcripts")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load video list
with open('/Users/qsy/Desktop/tennis/video_list.json', 'r') as f:
    videos = json.load(f)

print(f"Processing {len(videos)} videos...")

def fetch_transcript(video_id, lang='en'):
    """Fetch transcript using MCP server"""
    cmd = f'''cd {MCP_PATH} && node --input-type=module -e "
import {{ getSubtitles }} from './dist/youtube-fetcher.js';
try {{
    const result = await getSubtitles({{ videoID: '{video_id}', lang: '{lang}' }});
    console.log(JSON.stringify(result));
}} catch(e) {{
    console.log(JSON.stringify({{error: e.message}}));
}}
"'''
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.stdout:
            return json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e)}
    return {"error": "No output"}

# Process videos by section
results = {}
failed = []
processed = 0

for video in videos:
    section = video['section']
    title = video['title']
    vid = video['video_id']
    
    if section not in results:
        results[section] = []
    
    print(f"[{processed+1}/{len(videos)}] {title[:50]}... ", end="", flush=True)
    
    # Check if already processed
    transcript_file = OUTPUT_DIR / f"{vid}.json"
    if transcript_file.exists():
        print("(cached)")
        with open(transcript_file, 'r') as f:
            data = json.load(f)
    else:
        data = fetch_transcript(vid)
        if "error" not in data:
            # Save transcript
            with open(transcript_file, 'w') as f:
                json.dump(data, f, ensure_ascii=False)
            print("OK")
        else:
            print(f"FAILED: {data.get('error', 'unknown')[:30]}")
            failed.append(vid)
        time.sleep(0.5)  # Rate limiting
    
    # Extract key info
    if "error" not in data:
        transcript_text = ""
        if "lines" in data:
            transcript_text = " ".join([l.get("text", "") for l in data["lines"]])
        
        results[section].append({
            "title": title,
            "video_id": vid,
            "metadata": data.get("metadata", {}),
            "transcript_length": len(transcript_text),
            "transcript_preview": transcript_text[:500] if transcript_text else ""
        })
    
    processed += 1

# Save summary
summary_file = '/Users/qsy/Desktop/tennis/transcripts_summary.json'
with open(summary_file, 'w') as f:
    json.dump({
        "total_videos": len(videos),
        "processed": processed,
        "failed": len(failed),
        "failed_ids": failed,
        "sections": {k: len(v) for k, v in results.items()}
    }, f, ensure_ascii=False, indent=2)

print(f"\n✅ Done! Processed {processed} videos, {len(failed)} failed")
print(f"Summary saved to {summary_file}")
