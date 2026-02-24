
import json
import sys

def main():
    json_path = "learn_ytb/feeltennis_flat.json"
    target_ids = [
        "Q7Rejj-VXWA", # 5 Tips Accuracy
        "3D0dXKwwNe0", # Most Powerful Drill
        "ch09J2B1tSs", # 10 Fundamentals
        "tMWaDuZODDk", # Running Forehand
        "xzFTxKMtI_w", # Consistency Blueprint
        "RWWplT5yiRs"  # Inside Out
    ]
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        entries = []
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = data.get("entries", [])
            if not entries and "id" in data:
                entries = [data]
                
        with open("video_desc.txt", "w") as out:
            for entry in entries:
                if entry.get("id") in target_ids:
                    out.write(f"ID: {entry.get('id')}\n")
                    out.write(f"Title: {entry.get('title')}\n")
                    out.write(f"Description:\n{entry.get('description')}\n")
                    out.write("-" * 40 + "\n")
        print("Descriptions saved to video_desc.txt")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
