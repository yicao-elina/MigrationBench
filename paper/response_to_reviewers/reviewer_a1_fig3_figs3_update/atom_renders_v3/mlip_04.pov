#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic
  right -10.83*x up 14.64*y
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
atom(< -2.18,  -2.66,  -2.65>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #0
atom(< -0.12,  -2.66,  -6.21>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #1
atom(<  2.04,  -2.74,  -2.59>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #2
atom(< -4.23,  -2.66,  -6.21>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #3
atom(< -0.21,   0.61,  -1.28>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #4
atom(<  2.04,   0.61,  -5.19>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #5
atom(<  4.29,   0.61,  -1.28>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #6
atom(< -2.18,  -0.30,  -5.02>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #7
atom(< -0.11,  -6.16,  -1.35>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #8
atom(<  2.04,  -6.16,  -5.06>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #9
atom(<  4.18,  -6.16,  -1.35>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #10
atom(< -2.18,  -6.19,  -5.02>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #11
atom(< -2.18,   4.14,  -2.60>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #12
atom(< -0.08,   4.14,  -6.23>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #13
atom(<  2.04,   4.47,  -2.59>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #14
atom(< -4.27,   4.14,  -6.23>, 0.78, rgb <0.63, 0.55, 0.68>, 0.0, ase3) // #15
atom(< -2.18,   2.39,  -0.15>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #16
atom(< -0.20,   2.04,  -3.88>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #17
atom(<  2.04,   2.04,   0.00>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #18
atom(<  4.28,   2.04,  -3.88>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #19
atom(< -2.18,  -4.17,  -0.15>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #20
atom(< -0.08,  -4.33,  -3.81>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #21
atom(<  2.04,  -4.33,  -0.15>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #22
atom(< -4.28,  -4.33,  -3.81>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #23
atom(< -0.14,   6.09,  -1.33>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #24
atom(<  2.04,   6.09,  -5.10>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #25
atom(<  4.22,   6.09,  -1.33>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #26
atom(< -2.18,   5.94,  -5.02>, 0.88, rgb <0.29, 0.50, 0.72>, 0.0, ase3) // #27
atom(< -0.07,  -1.02,  -3.80>, 0.46, rgb <0.78, 0.16, 0.25>, 0.48, ase3) // #28
atom(<  0.25,  -0.65,  -3.62>, 0.46, rgb <0.78, 0.16, 0.25>, 0.48, ase3) // #29
atom(<  2.02,   0.21,  -2.60>, 0.46, rgb <0.78, 0.16, 0.25>, 0.48, ase3) // #30
atom(<  1.52,   1.13,  -2.88>, 0.46, rgb <0.78, 0.16, 0.25>, 0.48, ase3) // #31
atom(<  2.04,   2.05,  -2.59>, 1.10, rgb <0.86, 0.55, 0.05>, 0.0, ase3) // #32
cylinder {< -0.07,  -1.02,  -3.80>, <  0.09,  -0.84,  -3.71>, Rbond texture{pigment {color rgb <0.78, 0.16, 0.25> transmit 0.48} finish{ase3}}}
cylinder {<  0.25,  -0.65,  -3.62>, <  0.09,  -0.84,  -3.71>, Rbond texture{pigment {color rgb <0.78, 0.16, 0.25> transmit 0.48} finish{ase3}}}
cylinder {<  0.25,  -0.65,  -3.62>, <  1.13,  -0.22,  -3.11>, Rbond texture{pigment {color rgb <0.78, 0.16, 0.25> transmit 0.48} finish{ase3}}}
cylinder {<  2.02,   0.21,  -2.60>, <  1.13,  -0.22,  -3.11>, Rbond texture{pigment {color rgb <0.78, 0.16, 0.25> transmit 0.48} finish{ase3}}}
cylinder {<  2.02,   0.21,  -2.60>, <  1.77,   0.67,  -2.74>, Rbond texture{pigment {color rgb <0.78, 0.16, 0.25> transmit 0.48} finish{ase3}}}
cylinder {<  1.52,   1.13,  -2.88>, <  1.77,   0.67,  -2.74>, Rbond texture{pigment {color rgb <0.78, 0.16, 0.25> transmit 0.48} finish{ase3}}}
cylinder {<  1.52,   1.13,  -2.88>, <  1.78,   1.59,  -2.73>, Rbond texture{pigment {color rgb <0.78, 0.16, 0.25> transmit 0.48} finish{ase3}}}
cylinder {<  2.04,   2.05,  -2.59>, <  1.78,   1.59,  -2.73>, Rbond texture{pigment {color rgb <0.86, 0.55, 0.05> transmit 0.0} finish{ase3}}}
// no constraints
