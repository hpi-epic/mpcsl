#include "include/rcore.h"
#include "include/graph.h"
#include "include/matrix.h"

static void is_a_sink(int *a, int node, int *k, int nnodes, int *nbr,
    short int *matched) {

int j = 0;

  /* check whether the current node has outgoing arcs. */
  for (j = 0, *k = 0; j < nnodes; j++) {

    /* nodes that has satisfied the conditions and had their undirected arcs
     * changed into directed arcs should be ignored in later iterations, along
     * with any incident arcs. */
    if (matched[j] != 0)
      continue;

    if ((a[CMC(j, node, nnodes)] == 0) && (a[CMC(node, j, nnodes)] == 1)) {

      /* this node is not a candidate, go to the next one. */
      *k = -1;

      break;

    }/*THEN*/
    else if ((a[CMC(j, node, nnodes)] == 1) || (a[CMC(node, j, nnodes)] == 1)) {

      /* save adjacent nodes (connected by either an undirected or a directed
       * arc). */
      nbr[(*k)++] = j;

    }/*THEN*/

  }/*FOR*/

}/*IS_A_SINK*/

static int all_adjacent(int *a, int node, int k, int nnodes, int *nbr) {

int j = 0, l = 0;

  for (j = 0; j < k; j++) {

    /* for every node that is connected to the current node by an undirected
     * arc, we need to check that that node is adjacent to all other nodes
     * that are adjacent to the current node; the implication is that we can
     * skip nodes that are connected to the current node by a directed arc. */
    if ((a[CMC(nbr[j], node, nnodes)] == 0) ||
        (a[CMC(node, nbr[j], nnodes)] == 0))
      continue;

    for (l = 0; l < k; l++) {

      if (l == j)
        continue;

      if ((a[CMC(nbr[j], nbr[l], nnodes)] == 0) &&
          (a[CMC(nbr[l], nbr[j], nnodes)] == 0)) {

        /* this node violates the condition above. */
        return FALSE;

      }/*THEN*/

    }/*FOR*/

  }/*FOR*/

  return TRUE;

}/*ALL_ADJACENT*/

/* construct a consistent DAG extension of a CPDAG. */
SEXP pdag_extension(SEXP arcs, SEXP nodes, SEXP debug) {

int i = 0, j = 0, k = 0, t = 0, nnodes = length(nodes);
int changed = 0, left = nnodes;
int *a = NULL, *nbr = NULL;
bool debugging = isTRUE(debug);
short int *matched = NULL;
SEXP amat, result;

  /* build and dereference the adjacency matrix. */
  PROTECT(amat = arcs2amat(arcs, nodes));
  a = INTEGER(amat);

  /* allocate and initialize the neighbours and matched vectors. */
  nbr = Calloc1D(nnodes, sizeof(int));
  matched = Calloc1D(nnodes, sizeof(short int));

  for (t = 0; t < nnodes; t++) {

    if (debugging) {

      Rprintf("----------------------------------------------------------------\n");
      Rprintf("> performing pass %d.\n", t + 1);
      Rprintf("> candidate nodes: ");
        for (j = 0; j < nnodes; j++)
          if (matched[j] == 0)
            Rprintf("%s ", NODE(j));
      Rprintf("\n");

    }/*THEN*/

    for (i = 0; i < nnodes; i++) {

      /* if the node is already ok, skip it. */
      if (matched[i] != 0)
        continue;

      /* check whether the node is a sink (that is, whether is does not have
       * any child). */
      is_a_sink(a, i, &k, nnodes, nbr, matched);

      /* if the node is not a sink move on. */
      if (k == -1) {

        if (debugging)
          Rprintf("  * node %s is not a sink.\n", NODE(i));

        continue;

      }/*THEN*/
      else {

        if (debugging)
          Rprintf("  * node %s is a sink.\n", NODE(i));

      }/*ELSE*/

      if (!all_adjacent(a, i, k, nnodes, nbr)) {

        if (debugging)
          Rprintf("  * not all nodes linked to %s by an undirected arc are adjacent.\n", NODE(i));

        continue;

      }/*THEN*/
      else {

        if (debugging) {

          if (k == 0)
            Rprintf("  * no node is linked to %s by an undirected arc.\n", NODE(i));
          else
            Rprintf("  * all nodes linked to %s by an undirected arc are adjacent.\n", NODE(i));

        }/*THEN*/

      }/*ELSE*/

      /* the current node meets all the conditions, direct all the arcs towards it. */
      if (k == 0) {

        if (debugging)
          Rprintf("  @ no undirected arc to direct towards %s.\n", NODE(i));

      }/*THEN*/
      else {

        for (j = 0; j < k; j++)
          a[CMC(i, nbr[j], nnodes)] = 0;

        if (debugging)
          Rprintf("  @ directing all incident undirected arcs towards %s.\n", NODE(i));

      }/*ELSE*/

      /* set the changed flag. */
      changed = 1;

      /* exclude the node from later iterations. */
      matched[i] = 1;
      left--;

    }/*FOR*/

    /* if nothing changed in the last iteration or there are no more candidate
     * nodes, there is nothing else to do. */
    if ((changed == 0) || (left == 0))
      break;
    else
      changed = 0;

  }/*FOR*/

  /* build the new arc set from the adjacency matrix. */
  PROTECT(result = amat2arcs(amat, nodes));

  Free1D(nbr);
  Free1D(matched);
  UNPROTECT(2);

  return result;

}/*PDAG_EXTENSION*/

