
/* from cache.structure.c */
SEXP cache_structure(SEXP nodes, SEXP amat, SEXP debug);
SEXP cache_node_structure(int cur, SEXP nodes, int *amat, int nrow,
    int *status, bool debugging);
