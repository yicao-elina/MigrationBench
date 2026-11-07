# Extract selected complete extxyz frames by absolute_step without reading the
# full trajectory into the local release. Invoke with -v targets=step,step.
BEGIN {
    n = 0
    keep = 0
    buffer = ""
    split(targets, selected, ",")
    for (i in selected) wanted[selected[i]] = 1
}
{
    if (n == 0) {
        natoms = $1 + 0
        buffer = $0 ORS
        n = 1
        next
    }
    if (n == 1) {
        header = $0
        step_text = header
        sub(/^.*absolute_step=/, "", step_text)
        sub(/[^0-9].*$/, "", step_text)
        step = step_text + 0
        keep = (step in wanted)
        if (keep) buffer = buffer $0 ORS
        n = 2
        next
    }
    if (keep) buffer = buffer $0 ORS
    n++
    if (n == natoms + 2) {
        if (keep) printf "%s", buffer
        n = 0
        keep = 0
        buffer = ""
    }
}
