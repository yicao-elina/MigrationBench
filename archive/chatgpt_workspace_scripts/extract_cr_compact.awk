BEGIN { frame = -1; in_header = 0; atom = 0; cr = 0 }
$0 == "2050" { frame++; in_header = 1; atom = 0; cr = 0; next }
in_header { in_header = 0; next }
{
    atom++
    if ($1 == "Cr") {
        x[cr] = $2; y[cr] = $3; z[cr] = $4; cr++
    }
    if (atom == 2050) {
        printf "%d", base + frame * 200
        for (i = 0; i < cr; i++) printf " %.12g %.12g %.12g", x[i], y[i], z[i]
        printf "\n"
        delete x; delete y; delete z; cr = 0
    }
}
