>>> from test import testModels
>>> m = testModels.model1()
>>> m.findAllTransitions()
Tracing phase starting at x = [ 295.56323266  406.39105772] ; t = 0.0
Tracing minimum up
traceMinimum t0 = 0
....................................................................................................................
Tracing phase starting at x = [ 215.33138061 -149.94743491] ; t = 118.200482512
Tracing minimum down
traceMinimum t0 = 118.2
........................................................................................
Tracing minimum up
traceMinimum t0 = 118.2
......................................................................................................................................................................................................
Tracing phase starting at x = [ 0.0012022 -0.0009008] ; t = 223.70109097
Tracing minimum down
traceMinimum t0 = 223.701
...........................................
Tracing minimum up
traceMinimum t0 = 223.701
.....................................................
Tunneling from phase 1 to phase 0 at T=77.62
high_vev = [ 234.29584753 -111.48503794]
low_vev = [ 289.27692303  389.92589146]
Path deformation converged. 13 steps. fRatio = 6.33248e-03
Path deformation converged. 1 steps. fRatio = 4.90449e-03
Tunneling from phase 1 to phase 0 at T=77.62
high_vev = [ 234.29285578 -111.55553717]
low_vev = [ 289.27654927  389.9248891 ]
Path deformation converged. 13 steps. fRatio = 7.28449e-03
Path deformation converged. 1 steps. fRatio = 5.19015e-03
Tunneling from phase 1 to phase 0 at T=77.62
high_vev = [ 234.29000241 -111.62238024]
low_vev = [ 289.27617211  389.92389033]
Path deformation converged. 13 steps. fRatio = 8.20445e-03
Path deformation converged. 1 steps. fRatio = 5.64160e-03
Tunneling from phase 1 to phase 0 at T=77.62
high_vev = [ 234.28726641 -111.68611393]
low_vev = [ 289.27579573  389.92288753]
Path deformation converged. 13 steps. fRatio = 9.09575e-03
Path deformation converged. 1 steps. fRatio = 5.51904e-03
Tunneling from phase 1 to phase 0 at T=95.75
high_vev = [ 226.36380557 -146.61282486]
low_vev = [ 279.03846096  361.80201943]
Path deformation converged. 11 steps. fRatio = 1.64863e-02
Path deformation converged. 1 steps. fRatio = 1.36900e-02
Tunneling from phase 1 to phase 0 at T=79.42
high_vev = [ 233.14577417 -126.17676196]
low_vev = [ 288.57157228  388.04402235]
Path deformation converged. 11 steps. fRatio = 1.76587e-03
Path deformation converged. 1 steps. fRatio = 3.71998e-03
Tunneling from phase 1 to phase 0 at T=79.88
high_vev = [ 232.93529152 -127.75128567]
low_vev = [ 288.38495155  387.54485054]
Path deformation converged. 11 steps. fRatio = 3.17326e-03
Path deformation converged. 1 steps. fRatio = 3.07541e-03
Tunneling from phase 1 to phase 0 at T=79.99
high_vev = [ 232.88446672 -128.10659743]
low_vev = [ 288.3382847   387.41994228]
Path deformation converged. 11 steps. fRatio = 3.46447e-03
Path deformation converged. 1 steps. fRatio = 2.97274e-03
Tunneling from phase 1 to phase 0 at T=79.99
high_vev = [ 232.88288183 -128.11752535]
low_vev = [ 288.33682102  387.41602555]
Path deformation converged. 11 steps. fRatio = 3.47186e-03
Path deformation converged. 1 steps. fRatio = 2.96584e-03
Tunneling from phase 1 to phase 0 at T=79.99
high_vev = [ 232.88333316 -128.11442119]
low_vev = [ 288.33723614  387.41713999]
Path deformation converged. 11 steps. fRatio = 3.46976e-03
Path deformation converged. 1 steps. fRatio = 2.96662e-03

>>> # High-T transition (second-order):
... for key, val in m.TnTrans[0].iteritems():
...     if key is not 'instanton':
...         print key, ":", val
low_vev : [ 0.0666956  -0.05057576]
Delta_p : 0.0
Delta_rho : 0.0
high_phase : 2
crit_trans : {'low_vev': array([ 0.0666956 , -0.05057576]), 'Delta_rho': 0.0, 'high_phase': 2, 'instanton': None, 'high_vev': array([ 0.0666956 , -0.05057576]), 'action': 0.0, 'trantype': 2, 'Tcrit': 222.94128038031261, 'low_phase': 1}
Tnuc : 222.94128038
high_vev : [ 0.0666956  -0.05057576]
action : 0.0
trantype : 2
low_phase : 1

>>> # Low-T transition (first-order):
... for key, val in m.TnTrans[1].iteritems():
...     if key is not 'instanton':
...         print key, ":", val
low_vev : [ 288.33723614  387.41713999]
Delta_p : 114484820.963
Delta_rho : 360332331.162
crit_trans : {'low_vev': array([ 263.48801795,  314.65384215]), 'Delta_rho': 471749594.76531208, 'high_phase': 1, 'high_vev': array([ 220.02158042, -150.01483706]), 'low_phase': 0, 'Tcrit': 109.40840756819448, 'trantype': 1}
high_phase : 1
Tnuc : 79.9902708879
high_vev : [ 232.88333316 -128.11442119]
action : 11197.2875098
trantype : 1
low_phase : 0


