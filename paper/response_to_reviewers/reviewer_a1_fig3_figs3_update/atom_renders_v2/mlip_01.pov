#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic
  right -12.32*x up 13.58*y
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
atom(< -1.70,  -2.75,  -6.92>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #0
atom(< -0.18,   0.20,  -9.49>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #1
atom(<  2.42,  -1.89,  -6.92>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #2
atom(< -4.30,  -0.68,  -9.51>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #3
atom(<  0.18,  -1.29,  -4.14>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #4
atom(<  1.70,   1.66,  -6.71>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #5
atom(<  4.30,  -0.41,  -4.12>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #6
atom(< -2.43,   0.79,  -6.71>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #7
atom(<  1.08,  -5.78,  -8.71>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #8
atom(<  2.62,  -2.79, -11.32>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #9
atom(<  5.19,  -4.88,  -8.72>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #10
atom(< -1.53,  -3.68, -11.31>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #11
atom(< -2.61,   1.69,  -2.31>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #12
atom(< -1.07,   4.69,  -4.92>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #13
atom(<  1.53,   2.59,  -2.32>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #14
atom(< -5.19,   3.77,  -4.92>, 0.68, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #15
atom(< -2.00,  -1.33,  -2.02>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #16
atom(< -0.45,   1.67,  -4.54>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #17
atom(<  2.20,  -0.42,  -2.01>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #18
atom(<  3.69,   2.47,  -4.53>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #19
atom(< -1.05,  -5.69,  -6.45>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #20
atom(<  0.44,  -2.76,  -9.10>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #21
atom(<  2.99,  -4.84,  -6.46>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #22
atom(< -3.69,  -3.57,  -9.10>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #23
atom(< -0.64,   2.70,   0.00>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #24
atom(<  0.90,   5.69,  -2.61>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #25
atom(<  3.47,   3.60,  -0.01>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #26
atom(< -3.26,   4.80,  -2.61>, 0.78, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #27
atom(< -0.01,  -0.54,  -6.82>, 1.02, rgb <0.86, 0.55, 0.05>, 0.0, ase3) // #28

// no constraints
