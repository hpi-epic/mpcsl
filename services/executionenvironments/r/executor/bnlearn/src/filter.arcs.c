#include "include/rcore.h"
#include "include/graph.h"
#include "include/matrix.h"
#include "include/bn.h"

/* remove duplicate arcs from the arc set. */
SEXP unique_arcs(SEXP arcs, SEXP nodes, SEXP warn) {

  return c_unique_arcs(arcs, nodes, isTRUE(warn));

}/*UNIQUE_ARCS*/

/* C-level interface to unique_arcs. */
SEXP c_unique_arcs(SEXP arcs, SEXP nodes, bool warnlevel) {

int i = 0, k = 0, nrow = 0, uniq_rows = 0;
int *dup_arc = NULL;
SEXP result, arc_id, dup;

  /* the arc set is empty, nothing to do. */
  if (length(arcs) == 0)
    return arcs;

  /* there really is a non-empty arc set, process it. */
  nrow = length(arcs)/2;

  /* match the node labels in the arc set. */
  PROTECT(arc_id = arc_hash(arcs, nodes, FALSE, FALSE));
  /* check which are duplicated. */
  PROTECT(dup = duplicated(arc_id, FALSE));
  dup_arc = INTEGER(dup);

  /* count how many are not. */
  for (i = 0; i < nrow; i++)
    if (dup_arc[i] == 0)
      uniq_rows++;

  /* if there is no duplicate arc simply return the original arc set. */
  if (uniq_rows == nrow) {

    UNPROTECT(2);
    return arcs;

  }/*THEN*/

  /* warn the user if told to do so. */
  if (warnlevel > 0)
    warning("removed %d duplicate arcs.", nrow - uniq_rows);

  /* allocate and initialize the return value. */
  PROTECT(result = allocMatrix(STRSXP, uniq_rows, 2));

  /* store the correct arcs in the return value. */
  for (i = 0, k = 0; i < nrow; i++) {

    if (dup_arc[i] != 0)
      continue;

    SET_STRING_ELT(result, k, STRING_ELT(arcs, i));
    SET_STRING_ELT(result, k + uniq_rows, STRING_ELT(arcs, i + nrow));
    k++;

  }/*FOR*/

  /* allocate, initialize and set the column names. */
  setDimNames(result, R_NilValue, mkStringVec(2, "from", "to"));

  UNPROTECT(3);

  return result;

}/*C_UNIQUE_ARCS*/

/* determine which arcs are undirected. */
SEXP which_undirected(SEXP arcs, SEXP nodes) {

int i = 0, nrow = length(arcs)/2, nlvls = 0;
int *coords = NULL, *id = NULL;
SEXP result, labels, try, arc_id;

  /* get the node labels from the arcs, or use those passed down from R. */
  if (isNull(nodes))
    PROTECT(labels = unique(arcs));
  else
    labels = nodes;

  nlvls = length(labels);

  /* match the node labels in the arc set. */
  PROTECT(try = match(labels, arcs, 0));
  coords = INTEGER(try);

  /* initialize the checklist. */
  PROTECT(arc_id = allocVector(INTSXP, nrow));
  id = INTEGER(arc_id);

  /* fill the checklist with the UPTRI() coordinates, which uniquely
   * identify an arc modulo its direction. */
  for (i = 0; i < nrow; i++)
    id[i] = UPTRI(coords[i], coords[i + nrow], nlvls);

  PROTECT(result = dupe(arc_id));

  if (isNull(nodes))
    UNPROTECT(4);
  else
    UNPROTECT(3);

  return result;

}/*WHICH_UNDIRECTED*/

