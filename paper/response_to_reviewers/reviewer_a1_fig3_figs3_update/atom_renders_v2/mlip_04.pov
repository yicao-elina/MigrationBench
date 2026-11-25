#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic
  right -11.53*x up 13.60*y
  direction 1.00*z
  location <0,0,65.00> look_at <0,0,0>}
light_source {<-14.00,  10.00,  38.00> color White}

light_source {< 10.00, -14.00,  48.00> color White
  area_light <5.00, 0, 0>, <0, 5.00, 0>, 5, 5
  adaptive 1 jitter}
// no fog
#declare simple = finish {phong 0.7 ambient 0.4 diffuse 0.55}
#declare pale = finish {ambient 0.9 diffuse 0.30 roughness 0.001 specular 0.2 }
#declare intermediate = finish {ambient 0.4 diffuse 0.6 specular 0.1 roughness 0.04}
#declare vmd = finish {ambient 0.2 diffuse 0.80 phong 0.25 phong_size 10.0 specular 0.2 roughness 0.1}
#declare jmol = finish {ambient 0.4 diffuse 0.6 specular 1 roughness 0.001 metallic}
#declare ase2 = finish {ambient 0.2 brilliance 3 diffuse 0.6 metallic specular 0.7 roughness 0.04 reflection 0.15}
#declare ase3 = finish {ambient 0.4 brilliance 2 diffuse 0.6 metallic specular 1.0 roughness 0.001 reflection 0.0}
#declare glass = finish {ambient 0.4 diffuse 0.35 specular 1.0 roughness 0.001}
#declare glass2 = finish {ambient 0.3 diffuse 0.3 specular 1.0 reflection 0.25 roughness 0.001}
#declare Rcell = 0.000;
#declare Rbond = 0.000;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     translate LOC}
#end

// no cell vertices
atom(< -1.24,  -2.93,  -7.12>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #0
atom(<  0.25,  -0.04,  -9.64>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #1
atom(<  2.91,  -2.15,  -7.13>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #2
atom(< -3.77,  -0.90,  -9.64>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #3
atom(<  0.40,  -1.20,  -3.85>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #4
atom(<  2.03,   1.96,  -6.61>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #5
atom(<  4.81,  -0.27,  -3.85>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #6
atom(< -1.93,   0.34,  -7.13>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #7
atom(< -2.23,   1.74,  -2.28>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #8
atom(< -0.71,   4.68,  -4.85>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #9
atom(<  1.85,   2.84,  -2.03>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #10
atom(< -4.81,   3.81,  -4.85>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #11
atom(< -1.61,  -1.16,  -1.78>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #12
atom(< -0.18,   1.58,  -4.67>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #13
atom(<  2.58,  -0.64,  -1.93>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #14
atom(<  4.20,   2.51,  -4.67>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #15
atom(< -0.65,  -5.70,  -6.42>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #16
atom(<  0.89,  -2.85,  -9.12>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #17
atom(<  3.50,  -4.94,  -6.53>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #18
atom(< -3.22,  -3.72,  -9.12>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #19
atom(< -0.34,   2.63,   0.00>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #20
atom(<  1.24,   5.70,  -2.67>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #21
atom(<  3.93,   3.54,   0.00>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #22
atom(< -2.85,   4.66,  -2.72>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #23
atom(<  2.20,   1.16,  -3.75>, 1.02, rgb <0.86, 0.55, 0.05>, 0.0, ase3) // #24

// no constraints
