import numpy
import ctypes

name = "Human_Motion_Gen"
requires_callback = True
lib = "lib/libHuman_Motion_Gen.so"
lib_static = "lib/libHuman_Motion_Gen.a"
c_header = "include/Human_Motion_Gen.h"
nstages = 10

# Parameter             | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
params = \
[("lb"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (195,   1),  195),
 ("ub"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (195,   1),  195),
 ("x0"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (210,   1),  210),
 ("xinit"               , "dense" , ""               , ctypes.c_double, numpy.float64, ( 15,   1),   15),
 ("all_parameters"      , "dense" , ""               , ctypes.c_double, numpy.float64, (230,   1),  230),
 ("num_of_threads"      , "dense" , "solver_int32_unsigned", ctypes.c_uint  , numpy.uint32 , (  1,   1),    1)]

# Output                | Type    | Ctypes type    | Numpy type   | Shape     | Len
outputs = \
[("x01"                 , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x02"                 , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x03"                 , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x04"                 , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x05"                 , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x06"                 , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x07"                 , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x08"                 , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x09"                 , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x10"                 , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21)]

# Info Struct Fields
info = \
[('it', ctypes.c_int),
 ('it2opt', ctypes.c_int),
 ('res_eq', ctypes.c_double),
 ('res_ineq', ctypes.c_double),
 ('rsnorm', ctypes.c_double),
 ('rcompnorm', ctypes.c_double),
 ('pobj', ctypes.c_double),
 ('dobj', ctypes.c_double),
 ('dgap', ctypes.c_double),
 ('rdgap', ctypes.c_double),
 ('mu', ctypes.c_double),
 ('mu_aff', ctypes.c_double),
 ('sigma', ctypes.c_double),
 ('lsit_aff', ctypes.c_int),
 ('lsit_cc', ctypes.c_int),
 ('step_aff', ctypes.c_double),
 ('step_cc', ctypes.c_double),
 ('solvetime', ctypes.c_double),
 ('fevalstime', ctypes.c_double),
 ('solver_id', ctypes.c_int * 8)
]

# Dynamics dimensions
#   nvar    |   neq   |   dimh    |   dimp    |   diml    |   dimu    |   dimhl   |   dimhu    
dynamics_dims = [
	(21, 15, 0, 23, 6, 6, 0, 0), 
	(21, 15, 15, 23, 21, 21, 15, 0), 
	(21, 15, 15, 23, 21, 21, 15, 0), 
	(21, 15, 15, 23, 21, 21, 15, 0), 
	(21, 15, 15, 23, 21, 21, 15, 0), 
	(21, 15, 15, 23, 21, 21, 15, 0), 
	(21, 15, 15, 23, 21, 21, 15, 0), 
	(21, 15, 15, 23, 21, 21, 15, 0), 
	(21, 15, 15, 23, 21, 21, 15, 0), 
	(21, 0, 15, 23, 21, 21, 15, 0)
]