#include "include/rcore.h"
#include "include/globals.h"
#include "include/matrix.h"
#include "include/graph.h"
#include "include/learning.h"

void tabu_add(double *cache_value, int *ad, int *am, SEXP bestop, SEXP nodes,
    int *nnodes, int *from, int *to, double *max, SEXP tabu_list, int *cur,
    int *narcs, int *path, int *scratch, bool debugging);
void tabu_del(double *cache_value, int *w, int *am, SEXP bestop, SEXP nodes,
    int *nnodes, int *from, int *to, double *max, SEXP tabu_list, int *cur,
    int *narcs, bool debugging);
void tabu_rev(double *cache_value, int *b, int *am, SEXP bestop, SEXP nodes,
    int *nnodes, int *from, int *to, double *max, int *update, SEXP tabu_list,
    int *cur, int *narcs, double *mp, double *np, int *path, int *scratch,
    bool debugging);

/* create a numerical compact representation of a network structure (akin to
 * a hash, but not quite) to allow fast comparison and low memory usage. */
SEXP tabu_hash(SEXP amat, SEXP nodes, SEXP list, SEXP current) {

int *a = INTEGER(amat), nnodes = length(nodes);
SEXP hash;

  /* compute the hash. */
  PROTECT(hash = c_amat_hash(a, nnodes));

  /* set the hash into the tabu list. */
  SET_VECTOR_ELT(list, INT(current), hash);

  UNPROTECT(1);

  return hash;

}/*TABU_HASH*/

/* look up the current model (represented by the adjacency matrix) in the tabu list. */
int tabu_match(SEXP tabu_list, int *cur, int *amat, int *narcs, int *nnodes) {

int i = 0, j = 0, ntabu = length(tabu_list);
int *matches = NULL, *target = NULL;
SEXP tabu_network, hash;

  PROTECT(hash = c_amat_hash(amat, *nnodes));
  target = INTEGER(hash);

  for (i = 0; i < ntabu; i++) {

    /* extract the network from the tabu list. */
    tabu_network = VECTOR_ELT(tabu_list, (*cur + i) % ntabu);

    /* this element has not been initialized yet, skip. */
    if (isNull(tabu_network))
      continue;

    /* if it has a different number of arcs it can't be the same network, skip. */
    if (length(tabu_network) != *narcs)
      continue;

    /* it seems we really need to compare the two networks, so be it. */
    matches = INTEGER(tabu_network);

    /* if a network in the tabu list matches, return its index in the R-level list. */
    for (j = 0; j < *narcs; j++) {

      /* since the two array are ordered by construction, a simple
       * element-by-elements is all it's needed. */
      if (matches[j] != target[j])
        goto there;

    }/*FOR*/

    /* all arcs match, the two networks are the same; return the network's
     * index in the tabu list (R-style, counting from 1). */
    UNPROTECT(1);
    return (*cur + i) % ntabu + 1;

there:
    continue;

  }/*FOR*/

  UNPROTECT(1);

  return 0;

}/*TABU_MATCH*/

/* a single step of tabu search (one arc addition/removal/reversal, plus
 * lookup in the tabu list). */
SEXP tabu_step(SEXP amat, SEXP nodes, SEXP added, SEXP cache, SEXP reference,
    SEXP wlmat, SEXP blmat, SEXP tabu_list, SEXP current, SEXP baseline,
    SEXP nparents, SEXP maxp, SEXP debug) {

int nnodes = length(nodes), narcs = 0, i = 0, j = 0;
int *am = NULL, *ad = NULL, *w = NULL, *b = NULL;
int *cur = NULL, counter = 0, update = 1, from = 0, to = 0;
int *path = NULL, *scratch = NULL;
double *cache_value = NULL, max = NUM(baseline);
double *mp = REAL(maxp), *np = REAL(nparents);
bool debugging = isTRUE(debug);
SEXP bestop;

  /* allocate and initialize the return value (use FALSE as a canary value). */
  PROTECT(bestop = allocVector(VECSXP, 3));
  setAttrib(bestop, R_NamesSymbol, mkStringVec(3, "op", "from", "to"));

  /* allocate and initialize a dummy FALSE object. */
  SET_VECTOR_ELT(bestop, 0, ScalarLogical(FALSE));

  /* allocate buffers for c_has_path(). */
  path = Calloc1D(nnodes, sizeof(int));
  scratch = Calloc1D(nnodes, sizeof(int));

  /* save pointers to the numeric/integer matrices. */
  cache_value = REAL(cache);
  ad = INTEGER(added);
  am = INTEGER(amat);
  w = INTEGER(wlmat);
  b = INTEGER(blmat);
  cur = INTEGER(current);

  /* compute the number of arcs in the network. */
  for (i = 0; i < nnodes * nnodes; i++)
    if (am[i] > 0)
      narcs++;

  if (debugging) {

     /* count how may arcs are to be tested. */
     for (i = 0; i < nnodes * nnodes; i++)
       counter += ad[i];

     Rprintf("----------------------------------------------------------------\n");
     Rprintf("* trying to add one of %d arcs.\n", counter);

  }/*THEN*/

  /* test neighbours by arc addition. */
  tabu_add(cache_value, ad, am, bestop, nodes, &nnodes, &from, &to, &max,
    tabu_list, cur, &narcs, path, scratch, debugging);

  if (debugging) {

     /* count how may arcs are to be tested. */
     for (i = 0, counter = 0; i < nnodes * nnodes; i++)
       counter += am[i] * (1 - w[i]);

     Rprintf("----------------------------------------------------------------\n");
     Rprintf("* trying to remove one of %d arcs.\n", counter);

  }/*THEN*/

  /* test neighbours by arc deletion. */
  tabu_del(cache_value, w, am, bestop, nodes, &nnodes, &from, &to, &max,
    tabu_list, cur, &narcs, debugging);

  if (debugging) {

     /* count how may arcs are to be tested. */
     for (i = 0, counter = 0; i < nnodes; i++)
       for (j = 0; j < nnodes; j++)
         counter += am[CMC(i, j, nnodes)] * (1 - b[CMC(j, i, nnodes)]);

     Rprintf("----------------------------------------------------------------\n");
     Rprintf("* trying to reverse one of %d arcs.\n", counter);

  }/*THEN*/

  /* test neighbours by arc reversal. */
  tabu_rev(cache_value, b, am, bestop, nodes, &nnodes, &from, &to, &max,
    &update, tabu_list, cur, &narcs, mp, np, path, scratch, debugging);

  /* update the reference scores. */
  REAL(reference)[to] += cache_value[CMC(from, to, nnodes)];
  if (update == 2)
    REAL(reference)[from] += cache_value[CMC(to, from, nnodes)];

  Free1D(path);
  Free1D(scratch);

  UNPROTECT(1);

  return bestop;

}/*TABU_STEP*/

/* try to add an arc to the current network, minding the tabu list. */
void tabu_add(double *cache_value, int *ad, int *am, SEXP bestop, SEXP nodes,
    int *nnodes, int *from, int *to, double *max, SEXP tabu_list, int *cur,
    int *narcs, int *path, int *scratch, bool debugging) {

int i = 0, j = 0, idx = 0;
double temp = 0, tol = MACHINE_TOL;

  for (i = 0; i < *nnodes; i++) {

    for (j = 0; j < *nnodes; j++) {

      /* nothing to see, move along. */
      if (ad[CMC(i, j, *nnodes)] == 0)
        continue;

      /* retrieve the score delta from the cache. */
      temp = cache_value[CMC(i, j, *nnodes)];

      if (debugging) {

        Rprintf("  > trying to add %s -> %s.\n", NODE(i), NODE(j));
        Rprintf("    > delta between scores for nodes %s %s is %lf.\n",
          NODE(i), NODE(j), temp);

      }/*THEN*/

      /* this score delta is the best one at the moment, so add the arc if it
       * does not introduce cycles in the graph and does not match any of the
       * networks stored in the tabu list. */
      if (temp - *max > tol) {

        if (c_has_path(j, i, am, *nnodes, nodes, FALSE, FALSE, path, scratch,
              FALSE)) {

          if (debugging)
            Rprintf("    > not adding, introduces cycles in the graph.\n");

          continue;

        }/*THEN*/

        /* update the adjacency matrix. */
        am[CMC(i, j, *nnodes)] = 1;
        *narcs += 1;

        /* lookup in the tabu list. */
        idx = tabu_match(tabu_list, cur, am, narcs, nnodes);

        /* undo the changes in the adjacency matrix. */
        am[CMC(i, j, *nnodes)] = 0;
        *narcs -= 1;

        if (idx > 0) {

          if (debugging)
            Rprintf("    > not adding, network matches element %d in the tabu list.\n", idx);

          continue;

        }/*THEN*/

        if (debugging)
          Rprintf("    @ adding %s -> %s.\n", NODE(i), NODE(j));

        /* update the return value. */
        bestop_update(bestop, "set", NODE(i), NODE(j));
        /* store the node indices to update the reference scores. */
        *from = i;
        *to = j;

        /* update the threshold score delta. */
        *max = temp;

      }/*THEN*/

    }/*FOR*/

  }/*FOR*/

}/*TABU_ADD*/

/* try to delete an arc from the current network, minding the tabu list. */
void tabu_del(double *cache_value, int *w, int *am, SEXP bestop, SEXP nodes,
    int *nnodes, int *from, int *to, double *max, SEXP tabu_list, int *cur,
    int *narcs, bool debugging) {

int i = 0, j = 0, idx = 0;
double temp = 0, tol = MACHINE_TOL;

  for (i = 0; i < *nnodes; i++) {

    for (j = 0; j < *nnodes; j++) {

      /* nothing to see, move along. */
      if (am[CMC(i, j, *nnodes)] == 0)
        continue;

      /* whitelisted arcs are not to be removed, ever. */
      if (w[CMC(i, j, *nnodes)] == 1)
        continue;

      /* retrieve the score delta from the cache. */
      temp = cache_value[CMC(i, j, *nnodes)];

      if (debugging) {

        Rprintf("  > trying to remove %s -> %s.\n", NODE(i), NODE(j));
        Rprintf("    > delta between scores for nodes %s %s is %lf.\n",
          NODE(i), NODE(j), temp);

      }/*THEN*/

      if (temp - *max > tol) {

        /* update the adjacency matrix. */
        am[CMC(i, j, *nnodes)] = 0;
        *narcs -= 1;

        /* lookup in the tabu list. */
        idx = tabu_match(tabu_list, cur, am, narcs, nnodes);

        /* undo the changes in the adjacency matrix. */
        am[CMC(i, j, *nnodes)] = 1;
        *narcs += 1;

        if (idx > 0) {

          if (debugging > 0)
            Rprintf("    > not removing, network matches element %d in the tabu list.\n", idx);

          continue;

        }/*THEN*/

        if (debugging)
          Rprintf("    @ removing %s -> %s.\n", NODE(i), NODE(j));

        /* update the return value. */
        bestop_update(bestop, "drop", NODE(i), NODE(j));
        /* store the node indices to update the reference scores. */
        *from = i;
        *to = j;

        /* update the threshold score delta. */
        *max = temp;

      }/*THEN*/

    }/*FOR*/

  }/*FOR*/

}/*TABU_DEL*/

/* try to reverse an arc in the current network, minding the tabu list. */
void tabu_rev(double *cache_value, int *b, int *am, SEXP bestop, SEXP nodes,
    int *nnodes, int *from, int *to, double *max, int *update, SEXP tabu_list,
    int *cur, int *narcs, double *mp, double *np, int *path, int *scratch,
    bool debugging) {

int i = 0, j = 0, idx = 0;
double temp = 0, tol = MACHINE_TOL;

  for (i = 0; i < *nnodes; i++) {

    for (j = 0; j < *nnodes; j++) {

      /* nothing to see, move along. */
      if (am[CMC(i, j, *nnodes)] == 0)
        continue;

      /* don't reverse an arc if the one in the opposite direction is
       * blacklisted, ever. */
      if (b[CMC(j, i, *nnodes)] == 1)
        continue;

      /* do not reverse an arc if that means violating the limit on the
       * maximum number of parents. */
      if (np[i] >= *mp)
        continue;

      /* retrieve the score delta from the cache. */
      temp = cache_value[CMC(i, j, *nnodes)] + cache_value[CMC(j, i, *nnodes)];

      if (debugging) {

        Rprintf("  > trying to reverse %s -> %s.\n", NODE(i), NODE(j));
        Rprintf("    > delta between scores for nodes %s %s is %lf.\n",
          NODE(i), NODE(j), temp);

      }/*THEN*/

      if (temp - *max > tol) {

        if (c_has_path(i, j, am, *nnodes, nodes, FALSE, TRUE, path, scratch,
              FALSE)) {

          if (debugging)
            Rprintf("    > not reversing, introduces cycles in the graph.\n");

          continue;

        }/*THEN*/

        /* update the adjacency matrix. */
        am[CMC(i, j, *nnodes)] = 0;
        am[CMC(j, i, *nnodes)] = 1;

        /* lookup in the tabu list. */
        idx = tabu_match(tabu_list, cur, am, narcs, nnodes);

        /* undo the changes in the adjacency matrix. */
        am[CMC(i, j, *nnodes)] = 1;
        am[CMC(j, i, *nnodes)] = 0;

        if (idx > 0) {

          if (debugging)
            Rprintf("    > not reversing, network matches element %d in the tabu list.\n", idx);

          continue;

        }/*THEN*/

        if (debugging)
          Rprintf("    @ reversing %s -> %s.\n", NODE(i), NODE(j));

        /* update the return value. */
        bestop_update(bestop, "reverse", NODE(i), NODE(j));
        /* store the node indices to update the reference scores. */
        *from = i;
        *to = j;
        /* both nodes' reference scores must be updated. */
        *update = 2;

        /* update the threshold score delta. */
        *max = temp;

      }/*THEN*/

    }/*FOR*/

  }/*FOR*/

}/*TABU_REV*/
