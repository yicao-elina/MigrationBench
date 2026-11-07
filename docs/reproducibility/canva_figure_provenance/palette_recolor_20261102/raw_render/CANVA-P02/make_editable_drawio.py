from html import escape
from pathlib import Path


OUT = Path(__file__).with_name(
    "migrationbench_full_pipeline_two_distributions_three_models_editable.drawio"
)


edges = []
vertices = []


def xml_attr(value: str) -> str:
    return escape(str(value), quote=True)


def html_label(value: str) -> str:
    return xml_attr(value)


def vertex(cell_id, value, style, x, y, w, h, parent="1"):
    vertices.append(
        f'<mxCell id="{xml_attr(cell_id)}" value="{html_label(value)}" '
        f'style="{xml_attr(style)}" vertex="1" parent="{xml_attr(parent)}">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />'
        f'</mxCell>'
    )


def edge(
    cell_id,
    source=None,
    target=None,
    value="",
    style="",
    source_point=None,
    target_point=None,
    waypoints=None,
    parent="1",
):
    attrs = [
        f'id="{xml_attr(cell_id)}"',
        f'value="{html_label(value)}"',
        f'style="{xml_attr(style)}"',
        'edge="1"',
        f'parent="{xml_attr(parent)}"',
    ]
    if source is not None:
        attrs.append(f'source="{xml_attr(source)}"')
    if target is not None:
        attrs.append(f'target="{xml_attr(target)}"')
    geometry = ['<mxGeometry relative="1" as="geometry">']
    if source_point is not None:
        geometry.append(
            f'<mxPoint x="{source_point[0]}" y="{source_point[1]}" as="sourcePoint" />'
        )
    if target_point is not None:
        geometry.append(
            f'<mxPoint x="{target_point[0]}" y="{target_point[1]}" as="targetPoint" />'
        )
    if waypoints:
        geometry.append('<Array as="points">')
        for px, py in waypoints:
            geometry.append(f'<mxPoint x="{px}" y="{py}" />')
        geometry.append('</Array>')
    geometry.append('</mxGeometry>')
    edges.append(f'<mxCell {' '.join(attrs)}>{''.join(geometry)}</mxCell>')


def text_style(font_size=16, color="#173F5F", align="center", family="Arial"):
    return (
        f"text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;"
        f"fontFamily={family};fontSize={font_size};fontColor={color};"
        f"align={align};verticalAlign=middle;"
    )


def box_style(fill="#FFF5E6", stroke="#173F5F", font_size=16):
    return (
        "rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=2;"
        f"fontFamily=Arial;fontSize={font_size};fontColor=#173F5F;"
        "align=center;verticalAlign=middle;"
    )


def atom_style(fill, size=1.2):
    return (
        "ellipse;aspect=fixed;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor=#173F5F;strokeWidth={size};"
    )


def connector_style(arrow="block", width=1.4, color="#176A8A", dashed=False):
    return (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
        f"strokeColor={color};strokeWidth={width};endArrow={arrow};endFill=1;endSize=8;"
        + ("dashed=1;dashPattern=6 4;" if dashed else "")
    )


def straight_style(arrow="none", width=1.2, color="#173F5F"):
    return (
        "edgeStyle=none;rounded=0;html=1;"
        f"strokeColor={color};strokeWidth={width};endArrow={arrow};endFill=1;endSize=8;"
    )


def atom(cell_id, x, y, fill, size=1.2):
    vertex(cell_id, "", atom_style(fill, size), x, y, 20, 20)


def add_cluster(prefix, centers, fills, output_index=None):
    ids = []
    for idx, ((cx, cy), fill) in enumerate(zip(centers, fills)):
        cell_id = f"{prefix}_{idx}"
        atom(cell_id, cx - 10, cy - 10, fill)
        ids.append(cell_id)
    if output_index is not None:
        return ids[output_index]
    return ids[-1]


def add_model_glyph(prefix, y_center):
    input_centers = [(600, y_center - 20), (600, y_center), (600, y_center + 20)]
    hidden_centers = [(630, y_center - 10), (630, y_center + 10)]
    output_center = (670, y_center)
    in_ids = []
    hid_ids = []
    for idx, (cx, cy) in enumerate(input_centers):
        cell_id = f"{prefix}_in{idx + 1}"
        atom(cell_id, cx - 10, cy - 10, ["#1F77B4", "#F0A23B", "#1F77B4"][idx])
        in_ids.append(cell_id)
    for idx, (cx, cy) in enumerate(hidden_centers):
        cell_id = f"{prefix}_h{idx + 1}"
        atom(cell_id, cx - 10, cy - 10, "#F0A23B")
        hid_ids.append(cell_id)
    out_id = f"{prefix}_out"
    atom(out_id, output_center[0] - 10, output_center[1] - 10, "#7B3F98")
    for src in in_ids:
        for dst in hid_ids:
            edge(f"{prefix}_e_{src}_{dst}", src, dst, style=straight_style("none", 1.0, "#4A6070"))
    for src in hid_ids:
        edge(f"{prefix}_e_{src}_{out_id}", src, out_id, style=straight_style("none", 1.0, "#4A6070"))
    return in_ids, hid_ids, out_id


# Palette
navy = "#173F5F"
blue = "#1F77B4"
orange = "#F0A23B"
purple = "#7B3F98"
teal = "#176A8A"


# Title and section headings
vertex("title", "MigrationBench training design", text_style(26, "#111111"), 0, 10, 1500, 40)
vertex("head_coordinates", "Coordinates", text_style(22, "#111111"), 20, 70, 280, 40)
vertex("head_descriptors", "Descriptors", text_style(22, "#111111"), 330, 70, 280, 40)
vertex(
    "head_nn",
    "Neural Networks<br>for atomic species",
    text_style(20, "#111111"),
    640,
    65,
    360,
    60,
)
vertex("head_energy", "Total Energy", text_style(22, "#111111"), 1080, 70, 300, 40)


# Coordinate-distribution labels
vertex("coord_foundation", "Foundation", text_style(16, "#111111"), 20, 120, 130, 30)
vertex("coord_inhouse", "In-house", text_style(16, "#111111"), 160, 120, 130, 30)


# Coordinate axes: three real, editable arrowed axes for each distribution.
axis_style = straight_style("block", 1.2, navy)
for prefix, ox, oy in [("foundation", 90, 260), ("inhouse", 230, 260)]:
    edge(f"{prefix}_axis_r1", source_point=(ox, oy), target_point=(ox + 70, oy), style=axis_style)
    edge(f"{prefix}_axis_r2", source_point=(ox, oy), target_point=(ox, oy - 80), style=axis_style)
    edge(f"{prefix}_axis_r3", source_point=(ox, oy), target_point=(ox - 50, oy + 50), style=axis_style)
    vertex(f"{prefix}_r1", "R<sub>1</sub>", text_style(14, navy, "left", "Times New Roman"), ox + 55, oy + 5, 40, 25)
    vertex(f"{prefix}_r2", "R<sub>2</sub>", text_style(14, navy, "right", "Times New Roman"), ox - 45, oy - 95, 45, 25)
    vertex(f"{prefix}_r3", "R<sub>3</sub>", text_style(14, navy, "left", "Times New Roman"), ox - 60, oy + 45, 45, 25)


# Atom clusters in coordinate space.
add_cluster(
    "foundation_atoms",
    [(50, 225), (80, 205), (110, 225), (50, 255), (80, 285), (115, 255), (85, 245)],
    [blue, orange, blue, orange, blue, orange, purple],
    output_index=6,
)
add_cluster(
    "inhouse_atoms",
    [(200, 235), (220, 215), (245, 225), (195, 260), (225, 270), (255, 255), (235, 245)],
    [blue, orange, blue, orange, blue, orange, purple],
    output_index=6,
)


# Descriptor panel and its editable mathematical content.
vertex(
    "descriptor_panel",
    "",
    "rounded=1;arcSize=14;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
    f"strokeColor={navy};strokeWidth=2.4;",
    330,
    130,
    300,
    230,
)
descriptor_formula = (
    '<div style="font-family:Times New Roman;font-size:17px;line-height:1.45;text-align:left;">'
    'G<sup>pre</sup> = [G<sub>1</sub>, G<sub>2</sub>, ..., G<sub>N</sub>]<br>'
    'G<sup>pre</sup><sub>i</sub> = [G<sub>i</sub><sup>1</sup>, G<sub>i</sub><sup>2</sup>, G<sub>i</sub><sup>3</sup>, ...]'
    '<br><span style="font-size:20px;">⋮</span><br>'
    'G<sup>in</sup> = [G<sub>1</sub>, G<sub>2</sub>, ..., G<sub>m</sub>]<br>'
    'G<sup>in</sup><sub>i</sub> = [G<sub>i</sub><sup>1</sup>, G<sub>i</sub><sup>2</sup>, G<sub>i</sub><sup>3</sup>, ...]'
    '</div>'
)
vertex(
    "descriptor_formula",
    descriptor_formula,
    text_style(17, "#111111", "left", "Times New Roman"),
    15,
    20,
    270,
    190,
    parent="descriptor_panel",
)


# Main neural-network graph: 3 inputs, 4 + 4 hidden nodes, 1 output.
nn_layers = {
    "input": [(680, 190), (680, 240), (680, 290)],
    "hidden1": [(770, 160), (770, 210), (770, 260), (770, 310)],
    "hidden2": [(860, 160), (860, 210), (860, 260), (860, 310)],
    "output": [(950, 240)],
}
nn_ids = {}
for layer, coords in nn_layers.items():
    nn_ids[layer] = []
    for idx, (x, y) in enumerate(coords):
        nid = f"nn_{layer}_{idx + 1}"
        fill = blue if layer == "input" else orange if layer.startswith("hidden") else purple
        atom(nid, x, y, fill, 1.5)
        nn_ids[layer].append(nid)
for left, right in [("input", "hidden1"), ("hidden1", "hidden2"), ("hidden2", "output")]:
    for s_idx, src in enumerate(nn_ids[left]):
        for t_idx, dst in enumerate(nn_ids[right]):
            edge(
                f"nn_edge_{left}_{s_idx}_{right}_{t_idx}",
                src,
                dst,
                style=straight_style("none", 1.1, "#173F5F"),
            )
vertex("nn_layer_input", "input", text_style(12, "#111111"), 660, 335, 60, 25)
vertex("nn_layer_h1", "hidden 1", text_style(12, "#111111"), 740, 335, 70, 25)
vertex("nn_layer_h2", "hidden 2", text_style(12, "#111111"), 830, 335, 70, 25)
vertex("nn_layer_output", "output", text_style(12, "#111111"), 930, 335, 60, 25)


# Descriptor-to-network arrows, spread over the right boundary of the descriptor box.
for idx, target_id in enumerate(nn_ids["input"]):
    y_exit = [0.25, 0.5, 0.75][idx]
    edge(
        f"descriptor_to_nn_{idx + 1}",
        "descriptor_panel",
        target_id,
        style=straight_style("block", 1.4, teal) + f"exitX=1;exitY={y_exit};exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;",
    )


# Atomic energy contributions and total energy.
energy_ids = []
for idx, (x, label) in enumerate([(1050, "E<sub>1</sub>"), (1130, "E<sub>2</sub>"), (1210, "E<sub>i</sub>"), (1290, "E<sub>n</sub>")]):
    eid = f"energy_contribution_{idx + 1}"
    atom(eid, x, 240, purple, 1.3)
    vertex(f"energy_label_{idx + 1}", label, text_style(14, "#111111", "center", "Times New Roman"), x - 20, 270, 60, 25)
    energy_ids.append(eid)
edge("nn_to_energy_1", nn_ids["output"][0], energy_ids[0], style=straight_style("block", 1.4, teal))
for idx in range(len(energy_ids) - 1):
    edge(f"energy_chain_{idx + 1}", energy_ids[idx], energy_ids[idx + 1], style=straight_style("none", 1.1, teal))
vertex(
    "total_energy_sum",
    '<span style="font-family:Times New Roman;font-size:34px;">∑<sub>i</sub> E<sub>i</sub></span>',
    text_style(28, "#111111", "center", "Times New Roman"),
    1350,
    205,
    120,
    90,
)
edge("energy_to_total", energy_ids[-1], "total_energy_sum", style=straight_style("block", 1.4, teal))


# Divider between the two conceptual tiers.
edge(
    "tier_divider",
    source_point=(40, 390),
    target_point=(1460, 390),
    style=straight_style("none", 1.0, "#C4D0D7"),
)
vertex("head_training", "Training routes", text_style(22, "#111111"), 560, 400, 380, 40)


# Training-route source clusters.
foundation_data_out = add_cluster(
    "foundation_data",
    [(210, 470), (230, 450), (230, 490), (250, 470)],
    [blue, orange, orange, purple],
    output_index=3,
)
inhouse_data_out = add_cluster(
    "inhouse_data",
    [(210, 570), (230, 550), (230, 590), (250, 570)],
    [blue, orange, blue, purple],
    output_index=3,
)
finetune_data_out = add_cluster(
    "finetune_inhouse_data",
    [(200, 760), (220, 740), (220, 780), (240, 760)],
    [blue, orange, blue, purple],
    output_index=3,
)
vertex("route_foundation_data", "Foundation data", text_style(16, "#111111", "left"), 40, 440, 160, 30)
vertex("route_inhouse_data", "In-house data", text_style(16, "#111111", "left"), 40, 540, 160, 30)
vertex("route_checkpoint_label", "Foundation checkpoint", text_style(13, "#111111", "left"), 40, 640, 200, 25)
vertex("route_finetune_data_label", "In-house data", text_style(13, "#111111", "center"), 155, 795, 110, 25)


# Training boxes.
vertex("pretrain_box", "PRETRAIN", box_style(), 300, 450, 220, 70)
vertex("scratch_box", "SCRATCH", box_style(), 300, 550, 220, 70)
vertex("finetune_box", "FINE-TUNE", box_style(), 300, 680, 220, 70)


# Fine-tune checkpoint source and its two-way input path.
atom("foundation_checkpoint", 240, 650, purple, 1.5)
edge(
    "checkpoint_to_finetune",
    "foundation_checkpoint",
    "finetune_box",
    style=straight_style("block", 1.4, teal) + "exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.25;entryDx=0;entryDy=0;",
)
edge(
    "inhouse_to_finetune",
    finetune_data_out,
    "finetune_box",
    style=straight_style("block", 1.4, teal) + "exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.75;entryDx=0;entryDy=0;",
)


# Route model glyphs and their labels.
r1_in, r1_hid, r1_out = add_model_glyph("route_generalist", 485)
r2_in, r2_hid, r2_out = add_model_glyph("route_specialist", 585)
r3_in, r3_hid, r3_out = add_model_glyph("route_finetuned", 715)
vertex("route_generalist_label", "Foundation / Generalist", text_style(16, "#111111", "left"), 710, 440, 240, 30)
vertex("route_specialist_label", "Specialist model", text_style(16, "#111111", "left"), 710, 540, 200, 30)
vertex("route_finetuned_label", "Fine-tuned model", text_style(16, "#111111", "left"), 710, 670, 200, 30)


# Route arrows, with explicit ports to keep the three rows clean and parallel.
edge(
    "foundation_to_pretrain",
    foundation_data_out,
    "pretrain_box",
    style=straight_style("block", 1.4, teal) + "exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;",
)
edge(
    "pretrain_to_generalist",
    "pretrain_box",
    r1_in[1],
    style=straight_style("block", 1.4, teal) + "exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;",
)
edge(
    "inhouse_to_scratch",
    inhouse_data_out,
    "scratch_box",
    style=straight_style("block", 1.4, teal) + "exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;",
)
edge(
    "scratch_to_specialist",
    "scratch_box",
    r2_in[1],
    style=straight_style("block", 1.4, teal) + "exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;",
)
edge(
    "finetune_to_model",
    "finetune_box",
    r3_in[1],
    style=straight_style("block", 1.4, teal) + "exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;",
)


# One editable total-energy expression per training route.
route_sums = []
for idx, y in enumerate([450, 550, 680], 1):
    sid = f"route_sum_{idx}"
    vertex(
        sid,
        '<span style="font-family:Times New Roman;font-size:30px;">∑<sub>i</sub> E<sub>i</sub></span>',
        text_style(26, "#111111", "center", "Times New Roman"),
        980,
        y,
        120,
        70,
    )
    route_sums.append(sid)
edge("generalist_to_sum", r1_out, route_sums[0], style=straight_style("block", 1.4, teal))
edge("specialist_to_sum", r2_out, route_sums[1], style=straight_style("block", 1.4, teal))
edge("finetuned_to_sum", r3_out, route_sums[2], style=straight_style("block", 1.4, teal))


# Emit the editable draw.io document. Edges precede vertices so all lines sit beneath nodes.
header = '<?xml version="1.0" encoding="UTF-8"?>'
mxfile = (
    '<mxfile host="drawio" version="26.0.0" modified="2026-10-24T00:00:00.000Z" '
    'agent="Codex" type="device">'
)
model = (
    '<mxGraphModel dx="1600" dy="1000" grid="1" gridSize="10" guides="1" '
    'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
    'pageWidth="1500" pageHeight="900" math="1" shadow="0" background="#FFFFFF">'
)
root = ['<root>', '<mxCell id="0" />', '<mxCell id="1" parent="0" />']
root.extend(edges)
root.extend(vertices)
root.append('</root>')
document = "\n".join([header, mxfile, '<diagram name="MigrationBench training design">', model, *root, '</mxGraphModel>', '</diagram>', '</mxfile>', ''])
OUT.write_text(document, encoding="utf-8")
print(OUT)
