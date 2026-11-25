#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic
  right -10.68*x up 14.90*y
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
#declare Rbond = 0.075;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     translate LOC}
#end

// no cell vertices
atom(< -2.14,  -2.57,  -2.50>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #0
atom(< -0.06,  -2.57,  -6.12>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #1
atom(<  2.08,  -2.57,  -2.48>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #2
atom(< -4.22,  -2.58,  -6.11>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #3
atom(< -0.24,   0.21,  -1.19>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #4
atom(<  2.03,   0.21,  -5.12>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #5
atom(<  4.30,   0.22,  -1.19>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #6
atom(< -2.19,   0.20,  -4.94>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #7
atom(< -0.03,  -6.31,  -1.26>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #8
atom(<  2.08,  -6.31,  -4.91>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #9
atom(<  4.20,  -6.31,  -1.25>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #10
atom(< -2.13,  -6.32,  -4.90>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #11
atom(< -2.19,   4.06,  -2.55>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #12
atom(< -0.12,   4.06,  -6.13>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #13
atom(<  2.02,   4.01,  -2.51>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #14
atom(< -4.25,   4.06,  -6.13>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #15
atom(< -2.19,   2.45,  -0.07>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #16
atom(< -0.08,   1.94,  -3.72>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #17
atom(<  2.02,   1.96,  -0.08>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #18
atom(<  4.12,   1.96,  -3.72>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #19
atom(< -2.13,  -4.31,  -0.04>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #20
atom(< -0.06,  -4.33,  -3.71>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #21
atom(<  2.08,  -4.35,   0.00>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #22
atom(< -4.20,  -4.35,  -3.71>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #23
atom(< -0.07,   6.05,  -1.30>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #24
atom(<  2.03,   6.05,  -4.92>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #25
atom(<  4.10,   6.03,  -1.31>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #26
atom(< -2.18,   6.22,  -4.94>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #27
atom(< -0.06,  -1.16,  -3.71>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #28
atom(<  0.26,  -0.79,  -3.53>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #29
atom(<  2.03,   0.07,  -2.50>, 1.10, rgb <1.00, 0.87, 0.58>, 0.0, ase3) // #30
atom(<  1.53,   0.99,  -2.79>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #31
atom(<  2.05,   1.91,  -2.49>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #32
cylinder {< -0.06,  -1.16,  -3.71>, <  0.10,  -0.98,  -3.62>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  0.26,  -0.79,  -3.53>, <  0.10,  -0.98,  -3.62>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  0.26,  -0.79,  -3.53>, <  1.15,  -0.36,  -3.01>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  2.03,   0.07,  -2.50>, <  1.15,  -0.36,  -3.01>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.0} finish{ase3}}}
cylinder {<  2.03,   0.07,  -2.50>, <  1.78,   0.53,  -2.65>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.0} finish{ase3}}}
cylinder {<  1.53,   0.99,  -2.79>, <  1.78,   0.53,  -2.65>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  1.53,   0.99,  -2.79>, <  1.79,   1.45,  -2.64>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  2.05,   1.91,  -2.49>, <  1.79,   1.45,  -2.64>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
// no constraints
