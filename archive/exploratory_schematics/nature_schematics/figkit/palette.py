"""Single source of colour and type for every MigrationBench schematic."""

ATOM = {  # species: (base colour, drawn radius in Å)
    "Te": ("#c27aa6", 0.70),
    "Sb": ("#5f9ea9", 0.64),
    "Cr": ("#eba92c", 0.66),
}

INIT = {
    "foundation": ("#5fa86b", "Foundation", "zero-shot"),
    "finetune": ("#2a9d9a", "Fine-tuning", "foundation + target"),
    "scratch": ("#2f5f98", "Scratch", "specialist"),
}

THERMAL = {"600K": "600 K (single T)", "multiT": "Multi-T"}

TASK = {
    "static": ("#f5d3de", "Static"),
    "md": ("#efbdd0", "MD"),
    "within_gap": ("#e9a3c0", "In-gap"),
    "gap_to_ql": ("#df86ad", "Gap-to-QL"),
    "within_ql": ("#c97bb4", "Within-QL"),
    "sliding": ("#9b72b8", "Sliding"),
}
# Darker variants for text and lines drawn on white.
TASK_INK = {
    "static": "#b3687f",
    "md": "#b05d7d",
    "within_gap": "#b04f7e",
    "gap_to_ql": "#a8406f",
    "within_ql": "#8e4a85",
    "sliding": "#6a4790",
}

OPERATOR = {"fixed": ("#f2b3c8", "Fixed path"), "self": ("#e46aa2", "Self-relaxed")}
STATUS = {"complete": "#6aa88a", "failed": "#d0606a", "pending": "#b8bcc4"}

INK = "#1f2329"
MUTED = "#6b7280"
LINE = "#3a3f47"
FAINT = "#c9cdd4"
CONTAINER = "#f1f2f5"
CONTAINER_EDGE = "#e3e5ea"
GAP_TINT = "#f6e7c3"
WHITE = "#ffffff"

FONT = "Arial"
SIZE = {"letter": 8.0, "title": 7.0, "label": 6.0, "small": 5.2, "note": 4.6}
