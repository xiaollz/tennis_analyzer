import re,sys
def load(p):
    r=[]
    for l in open(p,encoding="utf-8"):
        m=re.match(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)",l)
        if m: r.append((int(m.group(1))*60+float(m.group(2)),m.group(3)))
    return r
def fmt(t): return f"{int(t//60):02d}:{t%60:04.1f}"
def excerpt(lines,windows):
    out=[]
    for tag,s,e in windows:
        out.append(f"### {tag}（{fmt(s)}–{fmt(e)}）")
        for t,x in lines:
            if s<=t<=e: out.append(f"- [{fmt(t)}] {x}")
        out.append("")
    return "\n".join(out)
def mm(s): a,b=s.split(":"); return int(a)*60+int(b)
