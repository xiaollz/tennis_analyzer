#!/usr/bin/env bash
set -euo pipefail

# Stable wrapper to run tennis_analyzer with the correct venv + local model.
# Designed for OpenClaw/Telegram usage where the user uploads videos via iCloud
# into the repo's drop folder.

REPO="/Users/qsy/Desktop/tennis"
PY="${REPO}/venv/bin/python"
MODEL="${REPO}/models/yolo11m-pose.pt"
INBOX="${REPO}/data/videos/video"
OUT_DIR="${REPO}/reports"

impact_mode="hybrid"     # "hybrid" | "pose"
impact_merge_s="0.8"
impact_audio_tol_frames="7"
confidence="0.5"
smooth="0.5"
slow="1.0"
device="auto"           # "auto" | "cpu" | "mps" | "cuda"
big3_ui="1"
metrics="0"             # default OFF to avoid numeric overlays
left_handed="0"
report="1"              # default ON (generate markdown report + charts)
report_dir=""
report_sample_fps="4.0"
view="auto"             # "auto" | "side" | "back"

input=""
output=""

usage() {
  cat <<'EOF'
Usage:
  run_tennis_analysis.sh --latest [options]
  run_tennis_analysis.sh <video-path-or-filename> [options]

Options:
  --latest                 Analyze newest .mp4 in /Users/qsy/Desktop/tennis/data/videos/video
  -i, --input PATH         Input video path (absolute or repo-relative)
  -o, --output PATH        Output video path (absolute or repo-relative)
  --impact-mode MODE       hybrid|pose (default: hybrid)
  --impact-merge-s S       Merge impacts within S seconds (default: 0.8)
  --impact-audio-tol N     Audio onset tolerance in video frames (hybrid only, default: 7)
  --device DEV             auto|cpu|mps|cuda (default: auto)
  --view VIEW              auto|side|back (default: auto)
  -c, --confidence VAL     Pose confidence threshold (default: 0.5)
  --smooth VAL             Keypoint smoothing 0-1 (default: 0.5)
  -s, --slow VAL           Slow motion factor (0-1] (default: 1.0)
  --big3-ui                Enable Big3 panel (default: on)
  --no-big3-ui             Disable Big3 panel
  --metrics                Enable numeric metrics overlay
  --no-metrics             Disable numeric metrics overlay (default: on)
  --report                 Generate report (default: on)
  --no-report              Disable report generation
  --report-dir DIR         Report output directory (default: <output>_report)
  --report-sample-fps FPS  Sample FPS for angle curves (default: 4)
  --left-handed            Player is left-handed
  -h, --help               Show this help
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --latest)
      input="__LATEST__"
      shift
      ;;
    -i|--input)
      input="${2:-}"
      shift 2
      ;;
    -o|--output)
      output="${2:-}"
      shift 2
      ;;
    --impact-mode)
      impact_mode="${2:-}"
      shift 2
      ;;
    --impact-merge-s)
      impact_merge_s="${2:-}"
      shift 2
      ;;
    --impact-audio-tol)
      impact_audio_tol_frames="${2:-}"
      shift 2
      ;;
    --device)
      device="${2:-}"
      shift 2
      ;;
    --view)
      view="${2:-}"
      shift 2
      ;;
    -c|--confidence)
      confidence="${2:-}"
      shift 2
      ;;
    --smooth)
      smooth="${2:-}"
      shift 2
      ;;
    -s|--slow)
      slow="${2:-}"
      shift 2
      ;;
    --big3-ui)
      big3_ui="1"
      shift
      ;;
    --no-big3-ui)
      big3_ui="0"
      shift
      ;;
    --metrics)
      metrics="1"
      shift
      ;;
    --no-metrics)
      metrics="0"
      shift
      ;;
    --left-handed)
      left_handed="1"
      shift
      ;;
    --report)
      report="1"
      shift
      ;;
    --no-report)
      report="0"
      shift
      ;;
    --report-dir)
      report_dir="${2:-}"
      shift 2
      ;;
    --report-sample-fps)
      report_sample_fps="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "${input}" ]]; then
        input="$1"
        shift
      else
        fail "Unknown arg: $1 (run with --help)"
      fi
      ;;
  esac
done

[[ -d "${REPO}" ]] || fail "Repo not found: ${REPO}"
[[ -x "${PY}" ]] || fail "venv python not found/executable: ${PY}"
[[ -f "${MODEL}" ]] || fail "Model not found: ${MODEL}"
mkdir -p "${OUT_DIR}"

if [[ -z "${input}" ]]; then
  usage
  fail "Missing input. Provide a video path/filename or use --latest."
fi

resolve_input() {
  local in="$1"
  local resolved=""

  if [[ "${in}" == "__LATEST__" ]]; then
    resolved="$(ls -t "${INBOX}"/*.mp4 2>/dev/null | head -n 1 || true)"
    [[ -n "${resolved}" ]] || fail "No .mp4 found in inbox: ${INBOX}"
    echo "${resolved}"
    return 0
  fi

  # If absolute path.
  if [[ "${in}" == /* ]]; then
    echo "${in}"
    return 0
  fi

  # If repo-relative path.
  if [[ -f "${REPO}/${in}" ]]; then
    echo "${REPO}/${in}"
    return 0
  fi

  # If just a filename, try inbox.
  if [[ -f "${INBOX}/${in}" ]]; then
    echo "${INBOX}/${in}"
    return 0
  fi

  # Fallback: return as-is (will fail later with clearer message).
  echo "${in}"
}

input="$(resolve_input "${input}")"
[[ -f "${input}" ]] || fail "Input video not found: ${input}"

# Basic "iCloud still downloading" guard: size must be > 0 bytes.
size_bytes="$(stat -f '%z' "${input}" 2>/dev/null || echo 0)"
[[ "${size_bytes}" -gt 0 ]] || fail "Input video size is 0 bytes (iCloud may still be downloading): ${input}"

if [[ -z "${output}" ]]; then
  stem="$(basename "${input}")"
  stem="${stem%.*}"
  ts="$(date +%Y%m%d_%H%M%S)"
  output="${OUT_DIR}/${stem}_combined_${impact_mode}_${ts}.mp4"
else
  # Resolve output relative to repo if needed.
  if [[ "${output}" != /* ]]; then
    output="${REPO}/${output}"
  fi
fi

# Ensure output directory exists (supports date-based folders).
mkdir -p "$(dirname "${output}")"

cmd=(
  "${PY}" -m tennis_analyzer.main
  "${input}"
  -o "${output}"
  -m "${MODEL}"
  -d "${device}"
  -c "${confidence}"
  --smooth "${smooth}"
  --slow "${slow}"
  --impact-mode "${impact_mode}"
  --impact-merge-s "${impact_merge_s}"
  --impact-audio-tol "${impact_audio_tol_frames}"
  --view "${view}"
)

if [[ "${big3_ui}" == "1" ]]; then
  cmd+=(--big3-ui)
fi
if [[ "${metrics}" == "0" ]]; then
  cmd+=(--no-metrics)
fi
if [[ "${left_handed}" == "1" ]]; then
  cmd+=(--left-handed)
fi
if [[ "${report}" == "1" ]]; then
  cmd+=(--report --report-sample-fps "${report_sample_fps}")
  if [[ -n "${report_dir}" ]]; then
    cmd+=(--report-dir "${report_dir}")
  fi
fi

echo "Input : ${input}"
echo "Output: ${output}"
echo "Model : ${MODEL}"
echo "Cmd   : ${cmd[*]}"
echo

cd "${REPO}"
"${cmd[@]}"

echo
echo "DONE: ${output}"
ls -lh "${output}" | awk '{print "Size:", $5}'

if [[ "${report}" == "1" ]]; then
  if [[ -n "${report_dir}" ]]; then
    rp="${report_dir}"
    [[ "${rp}" != /* ]] && rp="${REPO}/${rp}"
  else
    # Default behaviour: <output>_report
    rp="${output%.*}_report"
  fi
  echo "REPORT: ${rp}/report.md"
fi
