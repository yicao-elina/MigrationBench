#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic
  right -10.80*x up 14.76*y
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
atom(< -2.14,  -2.82,  -2.55>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #0
atom(< -0.04,  -2.82,  -6.19>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #1
atom(<  1.86,  -2.75,  -2.67>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #2
atom(< -4.27,  -2.83,  -6.21>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #3
atom(< -0.08,   0.63,  -1.01>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #4
atom(<  2.32,   0.63,  -5.18>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #5
atom(<  3.68,   0.49,  -1.62>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #6
atom(< -2.05,   0.59,  -4.92>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #7
atom(< -0.05,  -6.25,  -1.34>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #8
atom(<  2.06,  -6.25,  -4.99>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #9
atom(<  4.15,  -6.24,  -1.34>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #10
atom(< -2.25,  -6.24,  -5.04>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #11
atom(< -2.19,   4.13,  -2.48>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #12
atom(< -0.01,   4.13,  -6.27>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #13
atom(<  2.14,   4.06,  -2.51>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #14
atom(< -4.36,   4.03,  -6.26>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #15
atom(< -2.39,   2.31,  -0.25>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #16
atom(<  0.07,   2.05,  -3.70>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #17
atom(<  2.05,   2.29,   0.00>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #18
atom(<  4.26,   2.29,  -3.84>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #19
atom(< -2.16,  -4.43,  -0.12>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #20
atom(< -0.13,  -4.46,  -3.82>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #21
atom(<  2.05,  -4.36,  -0.12>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #22
atom(< -4.26,  -4.36,  -3.77>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #23
atom(< -0.01,   5.93,  -1.35>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #24
atom(<  2.07,   5.93,  -4.95>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #25
atom(<  4.17,   6.15,  -1.34>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #26
atom(< -2.24,   5.78,  -5.03>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #27
atom(< -0.06,  -1.10,  -3.78>, 0.46, rgb <0.78, 0.16, 0.25>, 0.48, ase3) // #28
atom(<  0.65,  -0.09,  -3.37>, 0.46, rgb <0.78, 0.16, 0.25>, 0.48, ase3) // #29
atom(<  1.35,  -0.18,  -2.96>, 1.10, rgb <0.86, 0.55, 0.05>, 0.0, ase3) // #30
atom(<  1.54,   1.27,  -2.85>, 0.46, rgb <0.78, 0.16, 0.25>, 0.48, ase3) // #31
atom(<  2.05,   1.97,  -2.56>, 0.46, rgb <0.78, 0.16, 0.25>, 0.48, ase3) // #32
cylinder {< -0.06,  -1.10,  -3.78>, <  0.29,  -0.60,  -3.57>, Rbond texture{pigment {color rgb <0.78, 0.16, 0.25> transmit 0.48} finish{ase3}}}
cylinder {<  0.65,  -0.09,  -3.37>, <  0.29,  -0.60,  -3.57>, Rbond texture{pigment {color rgb <0.78, 0.16, 0.25> transmit 0.48} finish{ase3}}}
cylinder {<  0.65,  -0.09,  -3.37>, <  1.00,  -0.14,  -3.17>, Rbond texture{pigment {color rgb <0.78, 0.16, 0.25> transmit 0.48} finish{ase3}}}
cylinder {<  1.35,  -0.18,  -2.96>, <  1.00,  -0.14,  -3.17>, Rbond texture{pigment {color rgb <0.86, 0.55, 0.05> transmit 0.0} finish{ase3}}}
cylinder {<  1.35,  -0.18,  -2.96>, <  1.44,   0.54,  -2.91>, Rbond texture{pigment {color rgb <0.86, 0.55, 0.05> transmit 0.0} finish{ase3}}}
cylinder {<  1.54,   1.27,  -2.85>, <  1.44,   0.54,  -2.91>, Rbond texture{pigment {color rgb <0.78, 0.16, 0.25> transmit 0.48} finish{ase3}}}
cylinder {<  1.54,   1.27,  -2.85>, <  1.79,   1.62,  -2.71>, Rbond texture{pigment {color rgb <0.78, 0.16, 0.25> transmit 0.48} finish{ase3}}}
cylinder {<  2.05,   1.97,  -2.56>, <  1.79,   1.62,  -2.71>, Rbond texture{pigment {color rgb <0.78, 0.16, 0.25> transmit 0.48} finish{ase3}}}
// no constraints
