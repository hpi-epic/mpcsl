#include "include/rcore.h"
#include "include/matrix.h"

#define BLANKET		 1
#define NEIGHBOUR	 2
#define PARENT		 3
#define CHILD		 4

SEXP cache_node_structure(int cur, SEXP nodes, int *amat, int nrow,
    int *status, bool debugging);

/* compute the cached values for all nodes. */
SEXP cache_structure(SEXP nodes, SEXP amat, SEXP debug) {

int i = 0, length_nodes = length(nodes);
int *status = NULL, *a = INTEGER(amat);
bool debugging = isTRUE(debug);

  SEXP bn, temp;

  /* allocate the list and set its attributes.*/
  PROTECT(bn = allocVector(VECSXP, length_nodes));
  setAttrib(bn, R_NamesSymbol, nodes);

  /* allocate and intialize the status vector. */
  status = Calloc1D(length_nodes, sizeof(int));

  if (isTRUE(debug))
    Rprintf("* (re)building cached information about network structure.\n");

  /* populate the list with nodes' data. */
  for (i = 0; i < length_nodes; i++) {

    /* (re)initialize the status vector. */
    memset(status, '\0', sizeof(int) * length_nodes);

    temp = cache_node_structure(i, nodes, a, length_nodes, status, debugging);

    /* save the returned list. */
    SET_VECTOR_ELT(bn, i, temp);

  }/*FOR*/

  UNPROTECT(1);

  Free1D(status);

  return bn;

}/*CACHE_STRUCTURE*/

/* compute the cached values for a single node (R-friendly). */
SEXP cache_partial_structure(SEXP nodes, SEXP target, SEXP amat, SEXP debug) {

int i = 0, length_nodes = length(nodes);
char *t = (char *)CHAR(STRING_ELT(target, 0));
int *status = NULL, *a = INTEGER(amat);
bool debugging = isTRUE(debug);
SEXP cached;

  if (isTRUE(debug))
    Rprintf("* (re)building cached information about node %s.\n", t);

  /* allocate and initialize the status vector. */
  status = Calloc1D(length_nodes, sizeof(int));

  /* iterate fo find the node position in the array.  */
  for (i = 0; i < length_nodes; i++)
    if (!strcmp(t, CHAR(STRING_ELT(nodes, i))))
      break;

  /* return the corresponding part of the bn structure. */
  cached = cache_node_structure(i, nodes, a, length_nodes, status, debugging);

  Free1D(status);

  return cached;

}/*CACHE_PARTIAL_STRUCTURE*/

/* backend to compute the cached values for a single node. */
SEXP cache_node_structure(int cur, SEXP nodes, int *amat, int nrow,
    int *status, bool debugging) {

int i = 0, j = 0;
int num_parents = 0, num_children = 0, num_neighbours = 0, num_blanket = 0;
SEXP structure, mb, nbr, children, parents;

  if (debugging)
    Rprintf("* node %s.\n", NODE(cur));

  for (i = 0; i < nrow; i++) {

    if (amat[CMC(cur, i, nrow)] == 1) {

      if (amat[CMC(i, cur, nrow)] == 0) {

        /* if a[i,j] = 1 and a[j,i] = 0, then i -> j. */
        if (debugging)
          Rprintf("  > found child %s.\n", NODE(i));

        status[i] = CHILD;

        /* check whether this child has any other parent. */
        for (j = 0; j < nrow; j++) {

          if ((amat[CMC(j, i, nrow)] == 1) && (amat[CMC(i, j, nrow)] == 0)
                && (j != cur)) {

            /* don't mark a neighbour as in the markov blanket. */
            if (status[j] <= 1) {

              status[j] = BLANKET;

              if (debugging)
                Rprintf("  > found node %s in markov blanket.\n", NODE(j));

            }/*THEN*/

          }/*THEN*/

        }/*FOR*/

      }/*THEN*/
      else {

        /* if a[i,j] = 1 and a[j,i] = 1, then i -- j. */
        if (debugging)
          Rprintf("  > found neighbour %s.\n", NODE(i));

        status[i] = NEIGHBOUR;

      }/*ELSE*/

    }/*THEN*/
    else {

      if (amat[CMC(i, cur, nrow)] == 1) {

        /* if a[i,j] = 0 and a[j,i] = 1, then i <- j. */
        if (debugging)
          Rprintf("  > found parent %s.\n", NODE(i));

        status[i] = PARENT;

      }/*THEN*/

    }/*ELSE*/

  }/*FOR*/

  /* count how may nodes fall in each category. */
  for (i = 0; i < nrow; i++) {

    switch(status[i]) {

      case CHILD:
        /* a child is also a neighbour and belongs into the markov blanket. */
        num_children++;
        num_neighbours++;
        num_blanket++;
        break;
      case PARENT:
        /* the same goes for a parent. */
        num_parents++;
        num_neighbours++;
        num_blanket++;
        break;
      case NEIGHBOUR:
        /* it's not known if this is parent or a children, but it's certainly a neighbour. */
        num_neighbours++;
        num_blanket++;
        break;
      case BLANKET:
        num_blanket++;
        break;
      default:
        /* this node is not even in the markov blanket. */
        break;

    }/*SWITCH*/

  }/*FOR*/

  if (debugging)
    Rprintf("  > node %s has %d parent(s), %d child(ren), %d neighbour(s) and %d nodes in the markov blanket.\n",
      NODE(cur), num_parents, num_children, num_neighbours, num_blanket);

  /* allocate the list and set its attributes. */
  PROTECT(structure = allocVector(VECSXP, 4));
  setAttrib(structure, R_NamesSymbol,
    mkStringVec(4, "mb", "nbr", "parents", "children"));

  /* allocate and fill the "children" element of the list. */
  PROTECT(children = allocVector(STRSXP, num_children));
  for (i = 0, j = 0; (i < nrow) && (j < num_children); i++)
    if (status[i] == CHILD)
      SET_STRING_ELT(children, j++, STRING_ELT(nodes, i));

  /* allocate and fill the "parents" element of the list. */
  PROTECT(parents = allocVector(STRSXP, num_parents));
  for (i = 0, j = 0; (i < nrow) && (j < num_parents); i++)
    if (status[i] == PARENT)
      SET_STRING_ELT(parents, j++, STRING_ELT(nodes, i));

  /* allocate and fill the "nbr" element of the list. */
  PROTECT(nbr = allocVector(STRSXP, num_neighbours));
  for (i = 0, j = 0; (i < nrow) && (j < num_neighbours); i++)
    if (status[i] >= NEIGHBOUR)
      SET_STRING_ELT(nbr, j++, STRING_ELT(nodes, i));

  /* allocate and fill the "mb" element of the list. */
  PROTECT(mb = allocVector(STRSXP, num_blanket));
  for (i = 0, j = 0; (i < nrow) && (j < num_blanket + num_neighbours); i++)
    if (status[i] >= BLANKET)
      SET_STRING_ELT(mb, j++, STRING_ELT(nodes, i));

  /* attach the string vectors to the list. */
  SET_VECTOR_ELT(structure, 0, mb);
  SET_VECTOR_ELT(structure, 1, nbr);
  SET_VECTOR_ELT(structure, 2, parents);
  SET_VECTOR_ELT(structure, 3, children);

  UNPROTECT(5);

  return structure;

}/*CACHE_NODE_STRUCTURE*/

