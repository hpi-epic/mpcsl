#include "include/rcore.h"
#include "include/data.frame.h"
#include "include/fitted.h"

/* in-place conversion of a list into a data frame. */
SEXP minimal_data_frame(SEXP obj) {

int n = length(VECTOR_ELT(obj, 0));
int *row = NULL;
SEXP rownames;

  /* generate and set the row names. */
  if (n > 0) {

    PROTECT(rownames = allocVector(INTSXP, 2));
    row = INTEGER(rownames);

    row[0] = NA_INTEGER;
    row[1] = -n;

  }/*THEN*/
  else {

    PROTECT(rownames = allocVector(INTSXP, 0));

  }/*ELSE*/

  setAttrib(obj, R_RowNamesSymbol, rownames);

  /* set the class name. */
  setAttrib(obj, R_ClassSymbol, mkString("data.frame"));

  UNPROTECT(1);

  return obj;

}/*MINIMAL_DATA_FRAME*/

/* efficient column extraction from data frames. */
SEXP dataframe_column(SEXP dataframe, SEXP name, SEXP drop) {

  return c_dataframe_column(dataframe, name, LOGICAL(drop)[0], FALSE);

}/*DATAFRAME_COLUMN*/

SEXP c_dataframe_column(SEXP dataframe, SEXP name, bool drop, bool keep_names) {

SEXP try, result, colnames = getAttrib(dataframe, R_NamesSymbol);
int *idx = NULL, nnames = length(name), name_type = TYPEOF(name);

  if (dataframe == R_NilValue)
    return R_NilValue;

  switch(name_type) {

    case STRSXP:

      /* column names passed as strings; match the corresponding indexes. */
      PROTECT(try = match(colnames, name, 0));
      idx = INTEGER(try);
      break;

    case REALSXP:

      /* these are almost good enough, coerce them to integers. */
      PROTECT(try = coerceVector(name, INTSXP));
      idx = INTEGER(try);
      break;

    case INTSXP:

      /* these are already indexes, nothing to do. */
      idx = INTEGER(name);
      break;

    default:

      error("this SEXP type is not handled in minimal.data.frame.column().");

  }/*SWITCH*/

  if ((nnames > 1) || (drop == 0)) {

    PROTECT(result = allocVector(VECSXP, nnames));

    for (int i = 0; i < nnames; i++)
      SET_VECTOR_ELT(result, i, VECTOR_ELT(dataframe, idx[i] - 1));

    if (keep_names)
      setAttrib(result, R_NamesSymbol, name);

    UNPROTECT(1);

  }/*THEN*/
  else {

    if (*idx != 0)
      result = VECTOR_ELT(dataframe, *idx - 1);
    else
      result = R_NilValue;

  }/*ELSE*/

  if (name_type != INTSXP)
    UNPROTECT(1);

  return result;

}/*C_DATAFRAME_COLUMN*/

/* create a data column from a standalone bn.fit.*node object. */
SEXP node2df(SEXP target, int n) {

fitted_node_e node_type = fitted_node_to_enum(target);
SEXP result, res_levels;

  switch(node_type) {

    case GNODE:
    case CGNODE:

      return allocVector(REALSXP, n);

    case DNODE:
    case ONODE:

      PROTECT(result = allocVector(INTSXP, n));
      memset(INTEGER(result), '\0', n * sizeof(int));
      if (node_type == ONODE)
        setAttrib(result, R_ClassSymbol, mkStringVec(2, "ordered", "factor"));
      else if (node_type == DNODE)
        setAttrib(result, R_ClassSymbol, mkString("factor"));
      res_levels = getAttrib(getListElement(target, "prob"), R_DimNamesSymbol);
      setAttrib(result, R_LevelsSymbol, VECTOR_ELT(res_levels, 0));

      UNPROTECT(1);

      return result;

    default:
      error("unknown node type (class: %s).",
         CHAR(STRING_ELT(getAttrib(target, R_ClassSymbol), 0)));

  }/*SWITCH*/

}/*NODE2DF*/

/* create a data column from a node in a fitted network. */
SEXP fitnode2df(SEXP fitted, SEXP node, int n) {

SEXP target = getListElement(fitted, (char *)CHAR(node));

  return node2df(target, n);

}/*FITNODE2DF*/

/* create a data frame from a fitted network. */
SEXP fit2df(SEXP fitted, int n) {

int i = 0, nnodes = length(fitted);
SEXP df, nodes;

  /* allocate the data frame. */
  PROTECT(nodes = getAttrib(fitted, R_NamesSymbol));
  PROTECT(df = allocVector(VECSXP, nnodes));
  /* fill the columns. */
  for (i = 0; i < nnodes; i++)
    SET_VECTOR_ELT(df, i, fitnode2df(fitted, STRING_ELT(nodes, i), n));
  /* set the column names. */
  setAttrib(df, R_NamesSymbol, nodes);
  /* add the labels to the return value. */
  minimal_data_frame(df);

  UNPROTECT(2);

  return df;

}/*FIT2DF*/

