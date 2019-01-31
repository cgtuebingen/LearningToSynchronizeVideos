%module dijkstra

%{
    #define SWIG_FILE_WITH_INIT
    #include "full.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* cost, int m, int n)}
%apply (int** ARGOUTVIEWM_ARRAY1, int* DIM1) {(int** tour, int* tour_len)}
%apply (int** ARGOUTVIEWM_ARRAY1, int* DIM1) {(int** box, int* box_len)}
%include "full.h"