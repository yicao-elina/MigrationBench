#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic
  right -10.70*x up 15.78*y
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
atom(< -2.01,  -3.24,  -2.53>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #0
atom(< -0.00,  -3.24,  -6.01>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #1
atom(<  2.01,  -3.24,  -2.53>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #2
atom(< -4.21,  -3.18,  -6.12>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #3
atom(< -0.00,  -0.11,  -1.37>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #4
atom(<  2.01,  -0.11,  -4.85>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #5
atom(<  4.21,  -0.17,  -1.26>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #6
atom(< -2.01,  -0.11,  -4.85>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #7
atom(< -0.00,  -6.73,  -1.20>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #8
atom(<  2.15,  -6.73,  -4.93>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #9
atom(<  4.21,  -6.66,  -1.26>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #10
atom(< -2.15,  -6.73,  -4.93>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #11
atom(<  0.00,   6.73,  -3.69>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #12
atom(< -2.15,   3.38,  -2.44>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #13
atom(< -0.00,   3.38,  -6.18>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #14
atom(<  2.15,   3.38,  -2.44>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #15
atom(< -4.21,   3.31,  -6.12>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #16
atom(< -2.17,   1.49,  -0.08>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #17
atom(< -0.00,   1.60,  -3.69>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #18
atom(<  2.17,   1.49,  -0.08>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #19
atom(<  4.21,   1.49,  -3.61>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #20
atom(< -2.04,  -4.85,   0.00>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #21
atom(<  0.00,  -4.95,  -3.69>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #22
atom(<  2.04,  -4.85,  -0.00>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #23
atom(< -4.21,  -4.85,  -3.77>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #24
atom(< -0.00,   5.25,  -1.20>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #25
atom(<  2.15,   5.25,  -4.93>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #26
atom(<  4.21,   5.44,  -1.26>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #27
atom(< -2.15,   5.25,  -4.93>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #28
atom(< -0.00,  -1.68,  -3.69>, 1.10, rgb <1.00, 0.87, 0.58>, 0.0, ase3) // #29
atom(<  0.71,  -0.67,  -3.28>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #30
atom(<  1.41,  -0.76,  -2.88>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #31
atom(<  1.60,   0.70,  -2.77>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #32
atom(<  2.11,   1.40,  -2.47>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #33
cylinder {< -0.00,  -1.68,  -3.69>, <  0.35,  -1.17,  -3.48>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.0} finish{ase3}}}
cylinder {<  0.71,  -0.67,  -3.28>, <  0.35,  -1.17,  -3.48>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  0.71,  -0.67,  -3.28>, <  1.06,  -0.71,  -3.08>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  1.41,  -0.76,  -2.88>, <  1.06,  -0.71,  -3.08>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  1.41,  -0.76,  -2.88>, <  1.50,  -0.03,  -2.82>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  1.60,   0.70,  -2.77>, <  1.50,  -0.03,  -2.82>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  1.60,   0.70,  -2.77>, <  1.85,   1.05,  -2.62>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  2.11,   1.40,  -2.47>, <  1.85,   1.05,  -2.62>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
// no constraints
