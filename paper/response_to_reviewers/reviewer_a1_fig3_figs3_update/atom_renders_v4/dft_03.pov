#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic
  right -10.78*x up 14.79*y
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
atom(< -2.11,  -2.83,  -2.48>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #0
atom(<  0.01,  -2.83,  -6.15>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #1
atom(<  2.10,  -2.85,  -2.49>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #2
atom(< -4.17,  -2.87,  -6.12>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #3
atom(<  0.03,   0.30,  -1.13>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #4
atom(<  2.25,   0.30,  -4.98>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #5
atom(<  4.00,   0.56,  -1.40>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #6
atom(< -2.09,   0.67,  -4.92>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #7
atom(<  0.02,  -6.25,  -1.27>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #8
atom(<  2.12,  -6.25,  -4.91>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #9
atom(<  4.25,  -6.26,  -1.26>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #10
atom(< -2.11,  -6.25,  -4.93>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #11
atom(< -2.19,   4.11,  -2.34>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #12
atom(<  0.09,   4.11,  -6.29>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #13
atom(<  2.18,   3.99,  -2.45>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #14
atom(< -4.35,   3.94,  -6.22>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #15
atom(< -2.25,   2.07,  -0.14>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #16
atom(< -0.20,   2.50,  -3.82>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #17
atom(<  2.08,   2.28,   0.00>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #18
atom(<  4.25,   2.28,  -3.76>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #19
atom(< -2.09,  -4.41,  -0.05>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #20
atom(< -0.00,  -4.40,  -3.71>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #21
atom(<  2.13,  -4.42,  -0.07>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #22
atom(< -4.21,  -4.42,  -3.68>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #23
atom(<  0.02,   5.91,  -1.27>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #24
atom(<  2.12,   5.91,  -4.91>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #25
atom(<  4.21,   6.16,  -1.28>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #26
atom(< -2.11,   5.68,  -4.93>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #27
atom(<  0.02,  -1.08,  -3.70>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #28
atom(<  0.72,  -0.08,  -3.29>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #29
atom(<  1.42,  -0.17,  -2.89>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #30
atom(<  1.62,   1.29,  -2.78>, 1.10, rgb <1.00, 0.87, 0.58>, 0.0, ase3) // #31
atom(<  2.12,   1.99,  -2.48>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #32
cylinder {<  0.02,  -1.08,  -3.70>, <  0.37,  -0.58,  -3.50>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  0.72,  -0.08,  -3.29>, <  0.37,  -0.58,  -3.50>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  0.72,  -0.08,  -3.29>, <  1.07,  -0.12,  -3.09>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  1.42,  -0.17,  -2.89>, <  1.07,  -0.12,  -3.09>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  1.42,  -0.17,  -2.89>, <  1.52,   0.56,  -2.83>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  1.62,   1.29,  -2.78>, <  1.52,   0.56,  -2.83>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.0} finish{ase3}}}
cylinder {<  1.62,   1.29,  -2.78>, <  1.87,   1.64,  -2.63>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.0} finish{ase3}}}
cylinder {<  2.12,   1.99,  -2.48>, <  1.87,   1.64,  -2.63>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
// no constraints
