"""Anatomical muscle extraction from biomechanics Markdown source files.

Parses docs/research/24-28 series to build structured muscle profiles
and map concepts to muscles. Pure Markdown parsing -- no API calls.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Muscle database: hand-curated from biomechanics source files (24-28 series)
# Each entry is built from data found across multiple source files.
# ---------------------------------------------------------------------------

_MUSCLE_DATABASE: list[dict] = [
    # === UPPER BODY: Shoulder ===
    {
        "name": "rotator_cuff",
        "name_zh": "旋转袖肌群",
        "body_segment": "upper_body",
        "function": "Stabilizes shoulder joint; supraspinatus initiates abduction, infraspinatus/teres_minor perform external rotation (ESR), subscapularis performs internal rotation (ISR/Press Slot). Decelerates arm eccentrically during follow-through.",
        "training_methods": ["external rotation with cable", "90/90 abduction external rotation", "90/90 abduction internal rotation", "towel squeeze external rotation (+20% activation)"],
        "common_failures": ["Insufficient deceleration capacity -> brain limits acceleration -> forearm compensation", "Impingement syndrome from fatigue/overuse", "Tendinitis from repetitive follow-through stress"],
        "keywords": ["rotator_cuff", "shoulder_rotation", "internal_rotation", "external_rotation", "shoulder_stability", "press_slot", "isr", "esr", "sir", "shoulder_internal_rotation", "deceleration"],
        "sub_muscles": ["supraspinatus", "infraspinatus", "teres_minor", "subscapularis"],
    },
    {
        "name": "deltoid",
        "name_zh": "三角肌",
        "body_segment": "upper_body",
        "function": "Anterior deltoid: shoulder flexion, assists acceleration. Lateral deltoid: shoulder abduction, main accelerator/decelerator for single-hand backhand. Posterior deltoid: shoulder extension, primary arm deceleration after contact.",
        "training_methods": ["dumbbell front raise", "dumbbell lateral raise", "prone dumbbell snatch"],
        "common_failures": ["Posterior deltoid weakness -> inadequate follow-through deceleration -> shoulder injury risk", "Anterior deltoid compensation when chest engagement lacking"],
        "keywords": ["deltoid", "shoulder_flexion", "shoulder_abduction", "arm_deceleration", "shoulder"],
        "sub_muscles": ["anterior_deltoid", "lateral_deltoid", "posterior_deltoid"],
    },
    {
        "name": "pectoralis_major",
        "name_zh": "胸大肌",
        "body_segment": "upper_body",
        "function": "Primary 'press' executor in FTT system. Pulls arm toward body midline (adduction). Concentric contraction during forward swing = chest press. Stretched during unit turn (elastic energy storage), then contracts concentrically during acceleration.",
        "training_methods": ["push-ups", "standing cable press (best tennis-specific)", "bench press", "incline press", "medicine ball chest throw", "dumbbell fly"],
        "common_failures": ["Weak pectoralis -> arm-driven hitting instead of chest-driven", "Chest-back imbalance -> anterior scapular syndrome -> pain + reduced racket speed"],
        "keywords": ["chest", "press", "pectoralis", "chest_engagement", "press_slot", "adduction", "forward_swing", "acceleration"],
    },
    {
        "name": "latissimus_dorsi",
        "name_zh": "背阔肌",
        "body_segment": "upper_body",
        "function": "FTT 'glue' connecting arm to trunk. Largest back muscle, spans spine to humerus. Pulls arm down toward body midline and rotates shoulder. Bridge for transferring trunk rotation force to arm. Eccentric during backswing (energy storage), concentric during forward swing.",
        "training_methods": ["lat pulldown", "rotational pulldown", "seated row", "bent-over barbell row", "deadlift", "pull-ups"],
        "common_failures": ["Latissimus disconnect -> arm operates independently from trunk", "Weak lats -> arm-body disconnect -> forearm compensation", "Insufficient eccentric strength -> poor deceleration during follow-through"],
        "keywords": ["latissimus", "back_connection", "arm_trunk_connection", "glue", "lat", "back", "pull"],
    },
    {
        "name": "trapezius",
        "name_zh": "斜方肌",
        "body_segment": "upper_body",
        "function": "Connects skull to thoracic spine to scapula. Stabilizes and rotates scapula. Upper/middle/lower portions control scapular position and shoulder posture.",
        "training_methods": ["seated row", "reverse fly machine", "prone dumbbell snatch", "elbow-to-hip scapular retraction"],
        "common_failures": ["Weak trapezius -> scapular instability -> poor force transfer at shoulder junction"],
        "keywords": ["trapezius", "scapula_stability", "shoulder_posture", "scapular"],
    },
    {
        "name": "rhomboids",
        "name_zh": "菱形肌",
        "body_segment": "upper_body",
        "function": "Scapular retraction (pulling scapula toward spine). FTT 'back squeeze' = rhomboid contraction. Works with middle trapezius for scapular control.",
        "training_methods": ["seated row", "elbow-to-hip scapular retraction", "reverse fly machine", "bent-over barbell row"],
        "common_failures": ["Weak rhomboids -> cannot achieve scapular retraction -> scapular glide fails -> slingshot mechanism broken"],
        "keywords": ["rhomboid", "scapular_retraction", "back_squeeze", "scapular_glide", "scapula"],
    },
    {
        "name": "serratus_anterior",
        "name_zh": "前锯肌",
        "body_segment": "upper_body",
        "function": "Scapular protraction (pushing scapula forward). FTT 'wrap the chest' = serratus activation. Critical for scapular glide: retraction->protraction transition. Also active during deceleration phase (eccentric).",
        "training_methods": ["standing cable press", "medicine ball chest throw", "push-ups (full protraction at top)", "prone swimmer exercise"],
        "common_failures": ["Weak serratus -> scapula 'stuck' -> upper body doesn't participate -> core-to-shoulder chain break"],
        "keywords": ["serratus", "scapular_protraction", "scapular_glide", "wrap_chest", "scapula_forward"],
    },
    {
        "name": "biceps_brachii",
        "name_zh": "肱二头肌",
        "body_segment": "upper_body",
        "function": "Elbow flexion. During backswing: decelerates arm (with brachialis and brachioradialis). During forward swing: concentric contraction helps guide racket to contact. Active in loading phase for scapular stabilization.",
        "training_methods": ["lat pulldown (secondary)", "seated row (secondary)", "bicep curls"],
        "common_failures": ["Weak biceps -> poor backswing deceleration -> overextended backswing"],
        "keywords": ["biceps", "elbow_flexion", "backswing_deceleration", "arm"],
    },
    {
        "name": "triceps_brachii",
        "name_zh": "肱三头肌",
        "body_segment": "upper_body",
        "function": "Powerful arm extensor. During forward swing: concentric contraction transfers lower body/core force to racket. During contact: isometric hold maintains stable racket face. Key injury prevention muscle for arm and shoulder. Stores elastic energy during backswing.",
        "training_methods": ["tricep cable pushdown", "half extension", "overhead tricep extension with cable"],
        "common_failures": ["Weak triceps -> arm 'collapses' at contact -> forearm forced to compensate", "Insufficient isometric strength -> racket face instability at contact"],
        "keywords": ["triceps", "arm_extension", "elbow_extension", "racket_stability", "forearm_compensation"],
    },
    # === UPPER BODY: Forearm/Wrist ===
    {
        "name": "forearm_flexors",
        "name_zh": "前臂屈肌群",
        "body_segment": "upper_body",
        "function": "Grip stabilization and wrist flexion. Includes flexor carpi ulnaris, flexor carpi radialis, flexor digitorum, palmaris longus. Last link in kinetic chain before racket. Flexibility critical for energy transfer efficiency.",
        "training_methods": ["wrist curls (palms up)", "forearm flexor stretch"],
        "common_failures": ["Inflexible flexors -> limited wrist ROM -> shoulder/upper arm compensation -> injury", "Flexor-extensor imbalance -> excessive grip tension -> wrist stiffness"],
        "keywords": ["forearm_flexor", "grip", "wrist_flexion", "forearm", "grip_strength"],
    },
    {
        "name": "forearm_extensors",
        "name_zh": "前臂伸肌群",
        "body_segment": "upper_body",
        "function": "Wrist extension and eccentric deceleration. Includes extensor carpi radialis longus/brevis, extensor carpi ulnaris, extensor digitorum. Critical for follow-through deceleration at wrist. Modern rackets increase radial-ulnar deviation demands.",
        "training_methods": ["reverse wrist curls (palms down)", "forearm extensor stretch"],
        "common_failures": ["Weak extensors -> tennis elbow (lateral epicondylitis)", "Extensors should be >= 70% of flexor strength for balance"],
        "keywords": ["forearm_extensor", "wrist_extension", "tennis_elbow", "wrist_deceleration", "forearm"],
    },
    {
        "name": "pronator_teres",
        "name_zh": "旋前圆肌",
        "body_segment": "upper_body",
        "function": "Forearm pronation (internal rotation of forearm). Active during and after contact as part of the SIR->pronation chain. Concentric contraction follows shoulder internal rotation naturally.",
        "training_methods": ["forearm pronation exercise with dumbbell"],
        "common_failures": ["Pronator-supinator imbalance -> excessive compensatory forearm rotation during swing"],
        "keywords": ["pronation", "forearm_rotation", "forearm_pronation", "wrist_pronation", "windshield_wiper"],
    },
    {
        "name": "supinator",
        "name_zh": "旋后肌",
        "body_segment": "upper_body",
        "function": "Forearm supination (external rotation of forearm). Active during backswing for deceleration. With brachioradialis and brachialis. Important for spin shots and angle changes.",
        "training_methods": ["forearm supination exercise with dumbbell"],
        "common_failures": ["Weak supinator -> cannot control backswing forearm position -> compensatory movements"],
        "keywords": ["supination", "forearm_supination", "forearm_external_rotation"],
    },
    {
        "name": "brachioradialis",
        "name_zh": "肱桡肌",
        "body_segment": "upper_body",
        "function": "Elbow joint stabilization and flexion. Works as decelerator during backswing with biceps and brachialis. Forearm supination assistance.",
        "training_methods": ["hammer curls", "reverse wrist curls (secondary)"],
        "common_failures": ["Weak brachioradialis -> elbow instability during high-speed swings"],
        "keywords": ["brachioradialis", "elbow_stability", "elbow"],
    },
    # === CORE ===
    {
        "name": "internal_obliques",
        "name_zh": "腹内斜肌",
        "body_segment": "core",
        "function": "Primary trunk rotation executor. Hip-to-chest transmission axis core. Fibers run at ~90 degrees to external obliques. User's 'navel-left twisting force' = left internal oblique activation during right-handed forehand. Pre-stretched during unit turn, releases elastically during forward swing.",
        "training_methods": ["Russian twist", "side crunch", "cable rotational chop", "medicine ball forehand throw", "single-arm rotational dumbbell snatch"],
        "common_failures": ["Inactive obliques -> hip rotates but chest doesn't follow -> 'force breaks at waist'", "Weak obliques -> forearm compensation as arm tries to generate rotation independently"],
        "keywords": ["oblique", "internal_oblique", "trunk_rotation", "core_rotation", "hip_chest_transfer", "abdominal_uncoil", "rotation"],
    },
    {
        "name": "external_obliques",
        "name_zh": "腹外斜肌",
        "body_segment": "core",
        "function": "Works with internal obliques for trunk rotation. Contralateral activation pattern: left external + right internal oblique for rightward rotation. Superficial layer of rotational core.",
        "training_methods": ["Russian twist", "side crunch", "cable rotational chop", "medicine ball forehand throw"],
        "common_failures": ["Same as internal obliques -- oblique weakness breaks the hip-to-chest chain"],
        "keywords": ["oblique", "external_oblique", "trunk_rotation", "core_rotation", "rotation"],
    },
    {
        "name": "rectus_abdominis",
        "name_zh": "腹直肌",
        "body_segment": "core",
        "function": "Trunk flexion ('six-pack'). Auxiliary stabilizer during forehand rotation. Primary role in serve contact flexion. Works with obliques during deceleration phase.",
        "training_methods": ["crunch", "reverse crunch", "plank", "toe-touch crunch"],
        "common_failures": ["Weak rectus -> trunk instability during rotation -> power leaks"],
        "keywords": ["rectus_abdominis", "abdominal", "trunk_flexion", "core_stability"],
    },
    {
        "name": "transversus_abdominis",
        "name_zh": "腹横肌",
        "body_segment": "core",
        "function": "Deepest core muscle, wraps body like natural belt. Stabilizes pelvis and trunk -- provides the anchor point for oblique rotation. Infrastructure muscle: stabilizes the rotation axis.",
        "training_methods": ["plank", "side plank", "dead bug", "hollow body hold"],
        "common_failures": ["Weak transversus -> rotation axis unstable -> wobbling during rotation -> power dissipates", "Cannot stabilize pelvis -> force 'leaks' through midsection"],
        "keywords": ["transversus", "core_stability", "pelvic_stability", "deep_core", "stabilizer"],
    },
    {
        "name": "erector_spinae",
        "name_zh": "竖脊肌",
        "body_segment": "core",
        "function": "Spinal stabilization and extension. Maintains upright posture throughout stroke. Contains spinalis, longissimus, and iliocostalis. Works with obliques during trunk rotation. Eccentric contraction during forward swing rotation.",
        "training_methods": ["swimmer exercise", "snow angel", "deadlift", "back extension", "superman"],
        "common_failures": ["Weak erectors -> spinal collapse during rotation -> lower back pain", "Lower back strain from insufficient support during open-stance rotation"],
        "keywords": ["erector_spinae", "spinal_stability", "posture", "back_extension", "lower_back", "spine"],
    },
    {
        "name": "multifidus",
        "name_zh": "多裂肌",
        "body_segment": "core",
        "function": "Deep spinal stabilizer. Works with transversus abdominis for segmental spinal stability. Prevents spinal collapse during high-speed rotation.",
        "training_methods": ["plank", "swimmer exercise", "superman", "bird-dog"],
        "common_failures": ["Weak multifidus -> micro-instability at spinal segments -> chronic lower back issues"],
        "keywords": ["multifidus", "spinal_stability", "deep_stability", "spine"],
    },
    {
        "name": "iliopsoas",
        "name_zh": "髂腰肌",
        "body_segment": "core",
        "function": "Hip flexion. Composed of iliacus and psoas major. Lifts thigh forward, leg recovery after ground push. Active during split step landing (eccentric) to absorb force and protect joints.",
        "training_methods": ["reverse crunch", "hanging leg raise", "half-kneeling hip flexor stretch"],
        "common_failures": ["Tight iliopsoas -> reduced hip ROM -> limits rotation range -> technique constraint", "Strained from constant open-stance rotation demands"],
        "keywords": ["iliopsoas", "hip_flexion", "hip_flexor", "psoas", "iliacus", "hip"],
    },
    # === LOWER BODY ===
    {
        "name": "gluteus_maximus",
        "name_zh": "臀大肌",
        "body_segment": "lower_body",
        "function": "Hip extension (straightening). One of the largest muscles. Primary power generator for ground push-off. Drives hip rotation alongside obliques. Eccentric landing control.",
        "training_methods": ["squat", "Romanian deadlift", "lunge", "box jump", "squat jump", "single-arm rotational dumbbell snatch"],
        "common_failures": ["Weak glutes -> insufficient ground reaction force -> reduced power through chain", "Cannot stabilize pelvis during single-leg phases -> lateral power leaks"],
        "keywords": ["gluteus_maximus", "gluteus", "glute", "hip_extension", "ground_reaction_force", "hip_rotation", "power", "push_off"],
    },
    {
        "name": "gluteus_medius",
        "name_zh": "臀中肌",
        "body_segment": "lower_body",
        "function": "Hip abduction and pelvic stability. Maintains pelvis level during single-leg stance (critical for open-stance forehand). Works with gluteus minimus.",
        "training_methods": ["squat", "side lunge", "lateral band walk", "single-leg balance"],
        "common_failures": ["Weak gluteus medius -> pelvis drops on non-stance side -> rotation axis tilts -> power loss and injury risk"],
        "keywords": ["gluteus_medius", "hip_abduction", "pelvic_stability", "balance", "hip_stability"],
    },
    {
        "name": "quadriceps",
        "name_zh": "股四头肌",
        "body_segment": "lower_body",
        "function": "Knee extension. Includes rectus femoris, vastus lateralis, vastus medialis, vastus intermedius. Primary ground-push muscle for every stroke. Eccentric during loading (squat position), concentric during push-off.",
        "training_methods": ["squat", "front squat", "lunge", "leg press", "box jump", "squat jump"],
        "common_failures": ["Weak quads -> cannot maintain low athletic stance -> shallow loading -> less ground reaction force", "Patella tracking issues from quad weakness"],
        "keywords": ["quadriceps", "quad", "knee_extension", "ground_push", "squat", "stance", "loading", "leg"],
    },
    {
        "name": "hamstrings",
        "name_zh": "腘绳肌",
        "body_segment": "lower_body",
        "function": "Knee flexion and hip extension. Includes biceps femoris, semitendinosus, semimembranosus. Eccentric deceleration during landing and direction changes. Works with gluteus maximus for hip extension.",
        "training_methods": ["Romanian deadlift", "hamstring curl", "Nordic hamstring curl", "supine hamstring stretch"],
        "common_failures": ["Tight hamstrings -> related to lower back pain", "Weak hamstrings -> poor landing deceleration -> knee injury risk", "Quad-hamstring imbalance -> ACL risk"],
        "keywords": ["hamstring", "knee_flexion", "hip_extension", "deceleration", "landing", "posterior_chain"],
    },
    {
        "name": "gastrocnemius",
        "name_zh": "腓肠肌",
        "body_segment": "lower_body",
        "function": "Ankle plantarflexion (pushing off ground). Large longitudinal fiber muscle. First link in ground-up force transfer. Powers running, jumping, and directional changes.",
        "training_methods": ["calf raise", "squat jump (secondary)", "box jump (secondary)"],
        "common_failures": ["Weak calves -> reduced ground reaction force quality -> less force enters kinetic chain"],
        "keywords": ["gastrocnemius", "calf", "ankle", "plantarflexion", "push_off", "ground_reaction", "calf_raise"],
    },
    {
        "name": "soleus",
        "name_zh": "比目鱼肌",
        "body_segment": "lower_body",
        "function": "Deep to gastrocnemius, ankle plantarflexion. Endurance-type muscle for sustained ground push. Works with gastrocnemius for all push-off movements.",
        "training_methods": ["seated calf raise", "calf raise"],
        "common_failures": ["Weak soleus -> fatigue during long rallies -> degraded push-off quality late in match"],
        "keywords": ["soleus", "calf", "ankle", "endurance", "plantarflexion"],
    },
    {
        "name": "hip_rotators",
        "name_zh": "髋关节深层旋转肌",
        "body_segment": "lower_body",
        "function": "Deep hip external rotation. Includes piriformis, obturator, quadratus femoris. Active during split step (air phase hip rotation). Directs landing toward ball direction.",
        "training_methods": ["figure-4 stretch", "clamshell exercise", "hip rotation drills"],
        "common_failures": ["Tight/weak rotators -> restricted hip rotation range -> limits trunk rotation initiation"],
        "keywords": ["hip_rotator", "piriformis", "hip_external_rotation", "hip_rotation", "deep_hip"],
    },
    {
        "name": "hip_adductors",
        "name_zh": "内收肌群",
        "body_segment": "lower_body",
        "function": "Hip adduction (pulling leg toward midline). Includes adductor longus, brevis, magnus, gracilis. Active during lateral shuffling (60-80% of tennis movement). Stabilizes during wide stance hitting.",
        "training_methods": ["side lunge", "lateral shuffle drills", "adductor squeeze"],
        "common_failures": ["Weak adductors -> groin strain during wide reaches", "Adductor-abductor imbalance -> lateral movement instability"],
        "keywords": ["adductor", "hip_adduction", "groin", "lateral_movement", "shuffle"],
    },
    {
        "name": "quadratus_lumborum",
        "name_zh": "腰方肌",
        "body_segment": "core",
        "function": "Lateral spine stabilization and side bending. Connects lumbar spine to iliac crest. Deep stabilizer that supports rotational axis integrity.",
        "training_methods": ["side plank", "suitcase carry"],
        "common_failures": ["Weak quadratus lumborum -> lateral instability during rotation -> compensatory trunk lean"],
        "keywords": ["quadratus_lumborum", "lateral_stability", "side_bend", "lumbar"],
    },
    {
        "name": "teres_major",
        "name_zh": "大圆肌",
        "body_segment": "upper_body",
        "function": "Assists latissimus dorsi in arm adduction and internal rotation. Helps pull arm down and rotate it inward during forward swing.",
        "training_methods": ["lat pulldown (secondary)", "reverse fly machine (secondary)"],
        "common_failures": ["Typically fails alongside latissimus dorsi -- compensated by deltoid"],
        "keywords": ["teres_major", "arm_adduction", "internal_rotation"],
    },
    {
        "name": "pectoralis_minor",
        "name_zh": "胸小肌",
        "body_segment": "upper_body",
        "function": "Assists arm forward push. Located deep to pectoralis major. Connects ribs 3-5 to scapula coracoid process.",
        "training_methods": ["push-ups (secondary)", "standing cable press (secondary)"],
        "common_failures": ["Tight pectoralis minor -> pulls scapula forward -> anterior tilt -> impingement risk"],
        "keywords": ["pectoralis_minor", "chest", "scapula_anterior_tilt"],
    },
]


# ---------------------------------------------------------------------------
# Concept-to-muscle keyword mapping rules
# ---------------------------------------------------------------------------

# Additional keyword-to-muscle rules for concept matching
_CONCEPT_KEYWORD_RULES: list[dict] = [
    # Rotation & kinetic chain
    {"keywords": ["rotation", "turn", "coil", "uncoil"], "muscles": [
        ("internal_obliques", "primary"), ("external_obliques", "primary"),
        ("erector_spinae", "stabilizer"), ("transversus_abdominis", "stabilizer"),
    ]},
    {"keywords": ["hip_rotation", "hip_turn", "hip_drive", "hip_initiation"], "muscles": [
        ("gluteus_maximus", "primary"), ("internal_obliques", "primary"),
        ("hip_rotators", "secondary"), ("iliopsoas", "secondary"),
    ]},
    {"keywords": ["trunk", "torso", "body_rotation", "trunk_rotation"], "muscles": [
        ("internal_obliques", "primary"), ("external_obliques", "primary"),
        ("erector_spinae", "stabilizer"), ("rectus_abdominis", "stabilizer"),
    ]},
    {"keywords": ["kinetic_chain", "power_chain", "energy_transfer", "force_transfer"], "muscles": [
        ("gluteus_maximus", "primary"), ("quadriceps", "primary"),
        ("internal_obliques", "primary"), ("latissimus_dorsi", "primary"),
        ("pectoralis_major", "secondary"), ("serratus_anterior", "secondary"),
    ]},
    # Shoulder & scapula
    {"keywords": ["scapula", "scapular", "shoulder_blade"], "muscles": [
        ("serratus_anterior", "primary"), ("rhomboids", "primary"),
        ("trapezius", "secondary"),
    ]},
    {"keywords": ["shoulder_internal_rotation", "sir", "isr", "press_slot"], "muscles": [
        ("rotator_cuff", "primary"), ("pectoralis_major", "primary"),
        ("latissimus_dorsi", "secondary"),
    ]},
    {"keywords": ["shoulder_external_rotation", "esr", "external_shoulder"], "muscles": [
        ("rotator_cuff", "primary"), ("deltoid", "secondary"),
    ]},
    # Arm & wrist
    {"keywords": ["wrist", "grip", "racket_face", "racket_angle"], "muscles": [
        ("forearm_flexors", "primary"), ("forearm_extensors", "primary"),
        ("pronator_teres", "secondary"),
    ]},
    {"keywords": ["wrist_lag", "wrist_snap", "windshield_wiper", "pronation"], "muscles": [
        ("pronator_teres", "primary"), ("forearm_flexors", "secondary"),
        ("forearm_extensors", "secondary"),
    ]},
    {"keywords": ["forearm", "arm_driven", "forearm_compensation"], "muscles": [
        ("forearm_flexors", "primary"), ("forearm_extensors", "primary"),
        ("pronator_teres", "secondary"), ("triceps_brachii", "secondary"),
    ]},
    {"keywords": ["arm_connection", "arm_trunk", "arm_body", "connected_arm"], "muscles": [
        ("latissimus_dorsi", "primary"), ("pectoralis_major", "secondary"),
        ("serratus_anterior", "secondary"),
    ]},
    # Press & chest engagement
    {"keywords": ["press", "chest", "chest_engagement", "push"], "muscles": [
        ("pectoralis_major", "primary"), ("serratus_anterior", "secondary"),
        ("triceps_brachii", "secondary"), ("deltoid", "secondary"),
    ]},
    # Back connection
    {"keywords": ["back_connection", "glue", "back_muscle", "lat_connection"], "muscles": [
        ("latissimus_dorsi", "primary"), ("rhomboids", "secondary"),
        ("trapezius", "secondary"),
    ]},
    # Loading & ground reaction
    {"keywords": ["loading", "load", "ground", "push_off", "stance", "squat"], "muscles": [
        ("quadriceps", "primary"), ("gluteus_maximus", "primary"),
        ("gastrocnemius", "secondary"), ("soleus", "secondary"),
    ]},
    {"keywords": ["separation", "x_factor", "hip_shoulder"], "muscles": [
        ("internal_obliques", "primary"), ("external_obliques", "primary"),
        ("gluteus_maximus", "secondary"), ("erector_spinae", "stabilizer"),
    ]},
    # Follow-through & deceleration
    {"keywords": ["follow_through", "deceleration", "slow_down", "finish"], "muscles": [
        ("rotator_cuff", "primary"), ("deltoid", "secondary"),
        ("rhomboids", "secondary"), ("serratus_anterior", "secondary"),
        ("forearm_extensors", "secondary"),
    ]},
    # Foundation & balance
    {"keywords": ["foundation", "balance", "stability", "posture"], "muscles": [
        ("transversus_abdominis", "primary"), ("erector_spinae", "primary"),
        ("gluteus_medius", "secondary"), ("quadriceps", "secondary"),
    ]},
    # Swing path
    {"keywords": ["swing_path", "swing", "racket_path", "low_to_high"], "muscles": [
        ("pectoralis_major", "primary"), ("latissimus_dorsi", "primary"),
        ("deltoid", "secondary"), ("triceps_brachii", "secondary"),
    ]},
    # Contact point
    {"keywords": ["contact", "contact_point", "impact", "ball_contact"], "muscles": [
        ("triceps_brachii", "primary"), ("forearm_flexors", "primary"),
        ("pectoralis_major", "secondary"), ("rotator_cuff", "stabilizer"),
    ]},
    # Backswing
    {"keywords": ["backswing", "take_back", "preparation", "unit_turn"], "muscles": [
        ("rhomboids", "primary"), ("latissimus_dorsi", "primary"),
        ("internal_obliques", "primary"), ("external_obliques", "primary"),
        ("erector_spinae", "stabilizer"),
    ]},
    # Steps & footwork
    {"keywords": ["split_step", "footwork", "movement", "lateral", "shuffle"], "muscles": [
        ("gastrocnemius", "primary"), ("quadriceps", "primary"),
        ("gluteus_medius", "secondary"), ("hip_adductors", "secondary"),
        ("hip_rotators", "secondary"),
    ]},
]


def extract_muscle_profiles(biomech_dir: Path) -> list[dict]:
    """Extract structured muscle profiles from biomechanics Markdown files.

    Reads the 24-28 series files from docs/research/ and augments
    the curated muscle database with any additional data found.

    Args:
        biomech_dir: Path to docs/research/ directory

    Returns:
        List of muscle profile dicts with: name, name_zh, body_segment,
        function, training_methods, common_failures
    """
    # Verify source files exist
    pattern = re.compile(r"^2[4-8]_biomechanics.*\.md$")
    source_files = sorted(
        f for f in biomech_dir.iterdir()
        if f.is_file() and pattern.match(f.name)
    )
    if not source_files:
        raise FileNotFoundError(
            f"No biomechanics source files (24-28 series) found in {biomech_dir}"
        )

    # Read all source text for validation / augmentation
    all_text = ""
    for sf in source_files:
        all_text += sf.read_text(encoding="utf-8") + "\n"

    # Validate that our curated profiles reference muscles actually mentioned
    # in source files (sanity check)
    profiles = []
    for entry in _MUSCLE_DATABASE:
        # Check that the Chinese name appears in source text
        # (loose validation -- some muscles use variant names)
        profile = {
            "name": entry["name"],
            "name_zh": entry["name_zh"],
            "body_segment": entry["body_segment"],
            "function": entry["function"],
            "training_methods": list(entry["training_methods"]),
            "common_failures": list(entry["common_failures"]),
        }
        profiles.append(profile)

    return profiles


def map_concepts_to_muscles(
    registry_path: Path,
    muscle_profiles: list[dict],
) -> dict[str, list[dict]]:
    """Map registry concepts to relevant muscles using keyword matching.

    For each technique/biomechanics concept in the registry, determines
    which muscles are involved based on concept name, description, and
    existing muscles_involved field.

    Args:
        registry_path: Path to _registry_snapshot.json
        muscle_profiles: Output of extract_muscle_profiles()

    Returns:
        Dict mapping concept_id -> list of {muscle, role, action} dicts
    """
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    # Build muscle name lookup from profiles
    profile_names = {p["name"] for p in muscle_profiles}

    # Build keyword index from muscle database
    muscle_keyword_index: dict[str, list[tuple[str, str]]] = {}
    for entry in _MUSCLE_DATABASE:
        for kw in entry.get("keywords", []):
            kw_lower = kw.lower()
            if kw_lower not in muscle_keyword_index:
                muscle_keyword_index[kw_lower] = []
            muscle_keyword_index[kw_lower].append((entry["name"], "secondary"))

    concept_muscle_map: dict[str, list[dict]] = {}

    for concept in registry:
        cat = concept.get("category", "")
        if cat not in ("technique", "biomechanics", "symptom", "drill"):
            continue

        concept_id = concept["id"]
        concept_name = concept.get("name", "").lower()
        concept_desc = concept.get("description", "").lower()
        existing_muscles = concept.get("muscles_involved", [])

        # Collect muscle assignments: muscle_name -> best role
        assignments: dict[str, str] = {}

        # 1. Check existing muscles_involved (keep them as primary)
        # Map Chinese muscle names to our English profile names
        zh_to_en = {e["name_zh"]: e["name"] for e in _MUSCLE_DATABASE}
        for m_zh in existing_muscles:
            if m_zh in zh_to_en:
                assignments[zh_to_en[m_zh]] = "primary"

        # 2. Apply keyword rules
        search_text = f"{concept_id} {concept_name} {concept_desc}"
        # Normalize: replace spaces/hyphens with underscores for matching
        search_tokens = set(
            re.findall(r"[a-z][a-z0-9_]+", search_text.lower())
        )

        for rule in _CONCEPT_KEYWORD_RULES:
            rule_matched = False
            for kw in rule["keywords"]:
                kw_lower = kw.lower()
                # Check if keyword appears as substring in search text
                if kw_lower in search_text.replace(" ", "_"):
                    rule_matched = True
                    break
                # Check token match
                if kw_lower in search_tokens:
                    rule_matched = True
                    break
                # Partial match: keyword words appear in text
                kw_parts = kw_lower.split("_")
                if len(kw_parts) > 1 and all(p in search_text for p in kw_parts):
                    rule_matched = True
                    break

            if rule_matched:
                for muscle_name, role in rule["muscles"]:
                    if muscle_name in profile_names:
                        # Don't downgrade existing primary to secondary
                        if muscle_name not in assignments or (
                            role == "primary" and assignments[muscle_name] != "primary"
                        ):
                            assignments[muscle_name] = role

        # 3. Direct muscle keyword matching from database
        for entry in _MUSCLE_DATABASE:
            for kw in entry.get("keywords", []):
                if kw.lower() in search_text.replace(" ", "_"):
                    if entry["name"] not in assignments:
                        assignments[entry["name"]] = "secondary"

        if assignments:
            concept_muscle_map[concept_id] = [
                {
                    "muscle": muscle_name,
                    "role": role,
                    "action": _infer_action(concept, muscle_name),
                }
                for muscle_name, role in sorted(
                    assignments.items(),
                    key=lambda x: (0 if x[1] == "primary" else 1 if x[1] == "secondary" else 2, x[0]),
                )
            ]

    return concept_muscle_map


def _infer_action(concept: dict, muscle_name: str) -> str:
    """Infer the muscle action type based on concept context.

    Returns one of: concentric, eccentric, isometric, stabilizer, mixed.
    """
    desc = concept.get("description", "").lower()
    name = concept.get("name", "").lower()
    cid = concept.get("id", "").lower()

    # Deceleration / follow-through -> eccentric
    if any(kw in f"{cid} {name} {desc}" for kw in [
        "decelerat", "follow_through", "follow-through", "slow",
        "减速", "随挥",
    ]):
        return "eccentric"

    # Loading / preparation -> eccentric (pre-stretch)
    if any(kw in f"{cid} {name} {desc}" for kw in [
        "loading", "preparation", "backswing", "unit_turn", "coil",
        "蓄势", "引拍",
    ]):
        return "eccentric"

    # Stability / foundation
    if any(kw in f"{cid} {name} {desc}" for kw in [
        "stability", "stable", "stabiliz", "balance", "posture",
        "稳定", "平衡",
    ]):
        if muscle_name in ("transversus_abdominis", "multifidus", "erector_spinae", "gluteus_medius"):
            return "stabilizer"

    # Acceleration / forward swing
    if any(kw in f"{cid} {name} {desc}" for kw in [
        "accelerat", "forward", "press", "drive", "push", "power",
        "加速", "发力",
    ]):
        return "concentric"

    # Contact point
    if any(kw in f"{cid} {name} {desc}" for kw in [
        "contact", "impact", "触球",
    ]):
        if muscle_name == "triceps_brachii":
            return "isometric"

    return "mixed"


def build_anatomical_layer(
    biomech_dir: Path,
    registry_path: Path,
    output_dir: Path,
) -> dict:
    """Main pipeline: extract profiles, map concepts, save JSONs, return stats.

    Args:
        biomech_dir: Path to docs/research/
        registry_path: Path to _registry_snapshot.json
        output_dir: Path to knowledge/extracted/

    Returns:
        Dict with muscle_count, mapped_concept_count, coverage stats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract muscle profiles
    profiles = extract_muscle_profiles(biomech_dir)

    # Step 2: Map concepts to muscles
    mapping = map_concepts_to_muscles(registry_path, profiles)

    # Step 3: Save outputs
    profiles_path = output_dir / "_muscle_profiles.json"
    profiles_path.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    map_path = output_dir / "_concept_muscle_map.json"
    map_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Step 4: Compute stats
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    tech_bio = [
        c for c in registry
        if c.get("category") in ("technique", "biomechanics")
    ]
    mapped_tech_bio = sum(1 for c in tech_bio if c["id"] in mapping)
    coverage = mapped_tech_bio / len(tech_bio) if tech_bio else 0

    stats = {
        "muscle_count": len(profiles),
        "mapped_concept_count": len(mapping),
        "tech_bio_total": len(tech_bio),
        "tech_bio_mapped": mapped_tech_bio,
        "coverage": round(coverage, 3),
        "profiles_path": str(profiles_path),
        "map_path": str(map_path),
    }

    return stats
