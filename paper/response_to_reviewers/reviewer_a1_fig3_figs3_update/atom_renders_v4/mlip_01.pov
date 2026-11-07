#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic
  right -10.77*x up 14.65*y
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
atom(< -2.13,  -2.42,  -2.46>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #0
atom(< -0.03,  -2.42,  -6.10>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #1
atom(<  2.08,  -2.44,  -2.45>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #2
atom(< -4.25,  -2.44,  -6.11>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #3
atom(<  0.01,   0.27,  -1.23>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #4
atom(<  2.11,   0.27,  -4.86>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #5
atom(<  4.23,   0.29,  -1.22>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #6
atom(< -2.11,   0.28,  -4.87>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #7
atom(< -0.04,  -6.19,  -1.22>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #8
atom(<  2.09,  -6.19,  -4.91>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #9
atom(<  4.17,  -6.18,  -1.25>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #10
atom(< -2.16,  -6.20,  -4.90>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #11
atom(< -2.10,   4.04,  -2.41>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #12
atom(<  0.03,   4.04,  -6.10>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #13
atom(<  2.14,   4.05,  -2.42>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #14
atom(< -4.18,   4.01,  -6.07>, 0.78, rgb <0.60, 0.58, 0.78>, 0.0, ase3) // #15
atom(< -2.13,   2.06,  -0.02>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #16
atom(<  0.01,   2.13,  -3.65>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #17
atom(<  2.17,   2.09,  -0.03>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #18
atom(<  4.23,   2.09,  -3.59>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #19
atom(< -2.10,  -4.22,  -0.00>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #20
atom(< -0.03,  -4.28,  -3.68>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #21
atom(<  2.03,  -4.23,   0.00>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #22
atom(< -4.25,  -4.23,  -3.73>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #23
atom(<  0.04,   6.08,  -1.18>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #24
atom(<  2.16,   6.08,  -4.86>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #25
atom(<  4.25,   6.10,  -1.20>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #26
atom(< -2.09,   6.07,  -4.86>, 0.88, rgb <0.61, 0.84, 0.83>, 0.0, ase3) // #27
atom(< -0.33,  -1.44,  -3.85>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #28
atom(< -0.01,  -1.07,  -3.67>, 1.10, rgb <1.00, 0.87, 0.58>, 0.0, ase3) // #29
atom(<  1.76,  -0.21,  -2.64>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #30
atom(<  1.26,   0.71,  -2.93>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #31
atom(<  1.78,   1.63,  -2.63>, 0.46, rgb <1.00, 0.87, 0.58>, 0.58, ase3) // #32
cylinder {< -0.33,  -1.44,  -3.85>, < -0.17,  -1.26,  -3.76>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {< -0.01,  -1.07,  -3.67>, < -0.17,  -1.26,  -3.76>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.0} finish{ase3}}}
cylinder {< -0.01,  -1.07,  -3.67>, <  0.87,  -0.64,  -3.15>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.0} finish{ase3}}}
cylinder {<  1.76,  -0.21,  -2.64>, <  0.87,  -0.64,  -3.15>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  1.76,  -0.21,  -2.64>, <  1.51,   0.25,  -2.79>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  1.26,   0.71,  -2.93>, <  1.51,   0.25,  -2.79>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  1.26,   0.71,  -2.93>, <  1.52,   1.17,  -2.78>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
cylinder {<  1.78,   1.63,  -2.63>, <  1.52,   1.17,  -2.78>, Rbond texture{pigment {color rgb <1.00, 0.87, 0.58> transmit 0.58} finish{ase3}}}
// no constraints
