#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic
  right -10.87*x up 14.83*y
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
atom(< -2.12,  -2.85,  -2.54>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #0
atom(< -0.04,  -2.85,  -6.15>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #1
atom(<  1.85,  -2.78,  -2.65>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #2
atom(< -4.28,  -2.86,  -6.19>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #3
atom(<  0.03,   0.60,  -0.99>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #4
atom(<  2.38,   0.60,  -5.07>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #5
atom(<  3.92,   0.55,  -1.46>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #6
atom(< -1.94,   0.49,  -4.84>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #7
atom(< -0.05,  -6.28,  -1.31>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #8
atom(<  2.06,  -6.28,  -4.97>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #9
atom(<  4.15,  -6.26,  -1.33>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #10
atom(< -2.27,  -6.26,  -5.03>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #11
atom(< -2.20,   4.11,  -2.45>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #12
atom(<  0.00,   4.11,  -6.27>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #13
atom(<  2.13,   4.03,  -2.49>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #14
atom(< -4.37,   3.97,  -6.25>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #15
atom(< -2.34,   2.18,  -0.21>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #16
atom(<  0.01,   2.27,  -3.72>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #17
atom(<  2.14,   2.24,   0.00>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #18
atom(<  4.29,   2.24,  -3.73>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #19
atom(< -2.17,  -4.45,  -0.11>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #20
atom(< -0.11,  -4.51,  -3.79>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #21
atom(<  2.04,  -4.39,  -0.12>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #22
atom(< -4.29,  -4.39,  -3.76>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #23
atom(< -0.02,   5.90,  -1.29>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #24
atom(<  2.10,   5.90,  -4.95>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #25
atom(<  4.14,   6.18,  -1.33>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #26
atom(< -2.23,   5.73,  -5.01>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #27
atom(< -0.07,  -1.12,  -3.76>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #28
atom(<  0.64,  -0.12,  -3.35>, 1.10, rgb <1.00, 0.87, 0.58>, 0.0, ase3) // #29
atom(<  1.34,  -0.21,  -2.95>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #30
atom(<  1.53,   1.25,  -2.84>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #31
atom(<  2.04,   1.95,  -2.55>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #32
cylinder {< -0.07,  -1.12,  -3.76>, <  0.28,  -0.62,  -3.56>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  0.64,  -0.12,  -3.35>, <  0.28,  -0.62,  -3.56>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.0} finish{ase3}}}
cylinder {<  0.64,  -0.12,  -3.35>, <  0.99,  -0.16,  -3.15>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.0} finish{ase3}}}
cylinder {<  1.34,  -0.21,  -2.95>, <  0.99,  -0.16,  -3.15>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  1.34,  -0.21,  -2.95>, <  1.43,   0.52,  -2.89>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  1.53,   1.25,  -2.84>, <  1.43,   0.52,  -2.89>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  1.53,   1.25,  -2.84>, <  1.78,   1.60,  -2.69>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  2.04,   1.95,  -2.55>, <  1.78,   1.60,  -2.69>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
// no constraints
