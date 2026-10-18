import sys, time
from faster_whisper import WhisperModel
src, out, model = sys.argv[1:4]
import wave, numpy as np
w=wave.open(src); src=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768.0
t0=time.time()
m = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=8)
segs, info = m.transcribe(src, language="en", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=700), condition_on_previous_text=False, beam_size=5)
with open(out, "w") as f:
    for s in segs:
        mi=int(s.start//60); se=s.start%60
        f.write(f"[{mi:02d}:{se:04.1f}] {s.text.strip()}\n"); f.flush()
f.close(); print("done", round(time.time()-t0), "s", "dur", round(len(src)/16000))
