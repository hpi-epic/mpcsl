
/* from data.frame.c */
SEXP minimal_data_frame(SEXP obj);
SEXP dataframe_column(SEXP dataframe, SEXP name, SEXP drop);
SEXP c_dataframe_column(SEXP dataframe, SEXP name, bool drop, bool keep_names);
SEXP node2df(SEXP target, int n);
SEXP fitnode2df(SEXP fitted, SEXP node, int n);
SEXP fit2df(SEXP fitted, int n);
