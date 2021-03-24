#include "include/rcore.h"
#include "include/sampling.h"
#include "include/sets.h"
#include "include/data.frame.h"
#include "include/graph.h"
#include "include/globals.h"
#include "include/fitted.h"

void rbn_discrete_root(SEXP result, int cur, SEXP cpt, int num, SEXP fixed);
void rbn_discrete_cond(SEXP result, SEXP nodes, int cur, SEXP parents, SEXP cpt,
    int num, SEXP fixed, bool debugging);
void rbn_gaussian(SEXP result, int cur, SEXP parents, SEXP coefs, SEXP sigma,
    int num, SEXP fixed);
void rbn_mixedcg(SEXP result, int cur, SEXP parents, SEXP coefs, SEXP sigma,
    SEXP dpar, SEXP gpar, int num, SEXP fixed);

/* generate random observations from a bayesian network. */
SEXP rbn_master(SEXP fitted, SEXP n, SEXP fix, SEXP debug) {

bool debugging = isTRUE(debug);
SEXP result;

  /* allocate the return value. */
  PROTECT(result = fit2df(fitted, INT(n)));
  /* perform the simulation. */
  c_rbn_master(fitted, result, n, fix, debugging);

  UNPROTECT(1);

  return result;

}/*RBN_MASTER*/

void c_rbn_master(SEXP fitted, SEXP result, SEXP n, SEXP fix, bool debugging) {

int num = INT(n), *poset = NULL, *mf = NULL;
int has_fixed = (TYPEOF(fix) != LGLSXP);
int i = 0, k = 0, cur = 0, nnodes = length(fitted), nparents = 0;
fitted_node_e cur_node_type = ENOFIT;
SEXP nodes, cur_node, cur_fixed, match_fixed, parents, parent_vars;
SEXP cpt = R_NilValue, coefs = R_NilValue, sd = R_NilValue;
SEXP dpar = R_NilValue, gpar = R_NilValue;

  /* allocate and initialize the return value. */
  PROTECT(nodes = getAttrib(fitted, R_NamesSymbol));

  /* order the nodes according to their depth in the graph. */
  poset = Calloc1D(nnodes, sizeof(int));
  topological_sort(fitted, poset, nnodes);

  /* match fixed nodes, if any, with the variables in the fitted network. */
  if (has_fixed) {

    PROTECT(match_fixed = match(getAttrib(fix, R_NamesSymbol), nodes, 0));
    mf = INTEGER(match_fixed);

  }/*THEN*/

  if (debugging) {

    Rprintf("* partial node ordering is:");

    for (i = 0; i < nnodes; i++)
      Rprintf(" %s", NODE(poset[i]));

    Rprintf(".\n");

  }/*THEN*/

  /* initialize the random number generator. */
  GetRNGstate();

  for (i = 0; i < nnodes; i++) {

    /* get the index of the node we have to generate random observations from,
     * its conditional probability table/regression parameters and the number
     * of its parents. */
    cur = poset[i];
    cur_node = VECTOR_ELT(fitted, cur);
    cur_node_type = fitted_node_to_enum(cur_node);
    parents = getListElement(cur_node, "parents");
    nparents = length(parents);

    /* check whether the value of the node is fixed, and if so retrieve it from
     * the list. */
    if (has_fixed && mf[cur] != 0)
      cur_fixed = VECTOR_ELT(fix, mf[cur] - 1);
    else
      cur_fixed = R_NilValue;

    /* find out whether the node corresponds to an ordered factor or not. */
    switch(cur_node_type) {

      case DNODE:
        cpt = getListElement(cur_node, "prob");
        break;

      case ONODE:
        cpt = getListElement(cur_node, "prob");
        break;

      case GNODE:
        coefs = getListElement(cur_node, "coefficients");
        sd = getListElement(cur_node, "sd");
        break;

      case CGNODE:
        coefs = getListElement(cur_node, "coefficients");
        sd = getListElement(cur_node, "sd");
        dpar = getListElement(cur_node, "dparents");
        gpar = getListElement(cur_node, "gparents");
        break;

      default:
        error("unknown node type (class: %s).",
           CHAR(STRING_ELT(getAttrib(cur_node, R_ClassSymbol), 0)));

    }/*SWITCH*/

    /* generate the random observations for the current node. */
    if (nparents == 0) {

      if (debugging) {

        if (cur_fixed != R_NilValue)
          Rprintf("* node %s is fixed.\n", NODE(cur));
        else
          Rprintf("* simulating node %s, which doesn't have any parent.\n",
            NODE(cur));

      }/*THEN*/

      switch(cur_node_type) {

        case DNODE:
          rbn_discrete_root(result, cur, cpt, num, cur_fixed);
          break;

        case ONODE:
          rbn_discrete_root(result, cur, cpt, num, cur_fixed);
          break;

        case GNODE:
          rbn_gaussian(result, cur, NULL, coefs, sd, num, cur_fixed);
          break;

        case CGNODE:
          /* this cannot happen, a conditional Gaussian node has at least
           * one discrete parent. */
          break;

        default:
          error("unknown node type (class: %s).",
             CHAR(STRING_ELT(getAttrib(cur_node, R_ClassSymbol), 0)));

      }/*SWITCH*/

    }/*THEN*/
    else {

      if (debugging) {

        if (cur_fixed != R_NilValue) {

          Rprintf("* node %s is fixed, ignoring parents.\n", NODE(cur));

        }/*THEN*/
        else {

          Rprintf("* simulating node %s with parents ", NODE(cur));
          for (k = 0; k < nparents - 1; k++)
            Rprintf("%s, ", CHAR(STRING_ELT(parents, k)));
          Rprintf("%s.\n", CHAR(STRING_ELT(parents, nparents - 1)));

        }/*ELSE*/

      }/*THEN*/

      PROTECT(parent_vars = dataframe_column(result, parents, FALSESEXP));

      switch(cur_node_type) {

        case DNODE:
          rbn_discrete_cond(result, nodes, cur, parent_vars, cpt, num,
            cur_fixed, debugging);
          break;

        case ONODE:
          rbn_discrete_cond(result, nodes, cur, parent_vars, cpt, num,
            cur_fixed, debugging);
          break;

        case GNODE:
          rbn_gaussian(result, cur, parent_vars, coefs, sd, num, cur_fixed);
          break;

        case CGNODE:
          rbn_mixedcg(result, cur, parent_vars, coefs, sd, dpar, gpar, num,
            cur_fixed);
          break;

        default:
          error("unknown node type (class: %s).",
             CHAR(STRING_ELT(getAttrib(cur_node, R_ClassSymbol), 0)));

      }/*SWITCH*/

      UNPROTECT(1);

    }/*ELSE*/

  }/*FOR*/

  PutRNGstate();

  Free1D(poset);

  UNPROTECT(1 + has_fixed);

}/*C_RBN_MASTER*/

void rbn_discrete_fixed(SEXP fixed, SEXP lvls, int *gen, int num) {

  if (length(fixed) == 1) {

    int constant = 0;

    /* fixed can be either a label to be matched with the factor's levels or
     * the corresponding numeric index, which can be used as it is. */
    if (TYPEOF(fixed) == INTSXP)
      constant = INT(fixed);
    else
      constant = INT(match(lvls, fixed, 0));

    for (int i = 0; i < num; i++)
      gen[i] = constant;

  }/*THEN*/
  else {

    SEXP fixed_levels;

    /* fixed is a set of labels here. */
    PROTECT(fixed_levels = match(lvls, fixed, 0));
    SampleReplace(num, length(fixed_levels), gen, INTEGER(fixed_levels));
    UNPROTECT(1);

  }/*ELSE*/

}/*RBN_DISCRETE_FIXED*/

/* unconditional discrete sampling. */
void rbn_discrete_root(SEXP result, int cur, SEXP cpt, int num, SEXP fixed) {

int np = length(cpt), *gen = NULL, *workplace = NULL;
double *p = NULL;
SEXP generated, lvls;

  /* get the levels of the curent variable .*/
  lvls = VECTOR_ELT(getAttrib(cpt, R_DimNamesSymbol), 0);
  /* get the column for the generated observations. */
  generated = VECTOR_ELT(result, cur);
  gen = INTEGER(generated);

  if (fixed != R_NilValue) {

    rbn_discrete_fixed(fixed, lvls, gen, num);

  }/*THEN*/
  else {

    workplace = Calloc1D(np, sizeof(int));

    /* duplicate the probability table to save the original copy from tampering. */
    p = Calloc1D(np, sizeof(double));
    memcpy(p, REAL(cpt), np * sizeof(double));

    /* perform the random sampling. */
    ProbSampleReplace(np, p, workplace, num, gen);

    Free1D(workplace);
    Free1D(p);

  }/*ELSE*/

}/*RBN_DISCRETE_ROOT*/

/* conditional discrete sampling. */
void rbn_discrete_cond(SEXP result, SEXP nodes, int cur, SEXP parents, SEXP cpt,
    int num, SEXP fixed, bool debugging) {

int np = length(cpt), nlevels = 0;
int *workplace = NULL, *configurations = NULL, *gen = NULL;
double *p = NULL;
bool warn = FALSE;
SEXP generated, lvls;

  /* get the number of levels of the curent variable .*/
  lvls = VECTOR_ELT(getAttrib(cpt, R_DimNamesSymbol), 0);
  nlevels = length(lvls);
  /* get the column for the generated observations. */
  generated = VECTOR_ELT(result, cur);
  gen = INTEGER(generated);

  if (fixed != R_NilValue) {

    rbn_discrete_fixed(fixed, lvls, gen, num);

  }/*THEN*/
  else {

    workplace = Calloc1D(np, sizeof(int));

    /* allocate and initialize the parents' configurations. */
    configurations = Calloc1D(num, sizeof(int));
    cfg(parents, configurations, NULL);

    /* duplicate the probability table to save the original copy from tampering. */
    p = Calloc1D(np, sizeof(double));
    memcpy(p, REAL(cpt), np * sizeof(double));
    /* perform the random sampling. */
    CondProbSampleReplace(nlevels, length(cpt)/nlevels, p, configurations,
      workplace, num, gen, &warn);

    Free1D(workplace);
    Free1D(configurations);
    Free1D(p);

  }/*ELSE*/

  /* warn when returning missing values. */
  if (warn && debugging)
    Rprintf("  > some parents configurations have undefined conditional distributions, NAs will be generated.");

}/*RBN_DISCRETE_COND*/

void rbn_gaussian_fixed(SEXP fixed, double *gen, int num) {

int i = 0;
double *constant = REAL(fixed);

  if (length(fixed) == 1) {

    /* conditioning on a single value. */
    for (i = 0; i < num; i++)
      gen[i] = constant[0];

  }/*THEN*/
  else {

    double offset = constant[0], range = constant[1] - constant[0];

    /* conditioning on an interval, picking a value at random
     * from a uniform distribution. */
    for (i = 0; i < num; i++)
      gen[i] = offset + unif_rand() * range;

  }/*ELSE*/

}/*RBN_GAUSSIAN_FIXED*/

/* conditional and unconditional normal sampling. */
void rbn_gaussian(SEXP result, int cur, SEXP parents, SEXP coefs, SEXP sigma,
    int num, SEXP fixed) {

int i = 0, j = 0, p = length(coefs);
double *beta = REAL(coefs), *sd = REAL(sigma), *gen = NULL, *Xj = NULL;
SEXP generated;

  /* get the column for the generated observations. */
  generated = VECTOR_ELT(result, cur);
  gen = REAL(generated);

  if (fixed != R_NilValue) {

    rbn_gaussian_fixed(fixed, gen, num);

  }/*THEN*/
  else {

    /* initialize with intercept and standard error. */
    for (i = 0; i < num; i++)
      gen[i] = beta[0] + norm_rand() * (*sd);

    /* add the contributions of the other regressors (if any). */
    for (j = 1; j < p; j++) {

      Xj = REAL(VECTOR_ELT(parents, j - 1));

      for (i = 0; i < num; i++)
        gen[i] += Xj[i] * beta[j];

    }/*FOR*/

  }/*ELSE*/

}/*RBN_GAUSSIAN*/

/* conditional linear Gaussian sampling. */
void rbn_mixedcg(SEXP result, int cur, SEXP parents, SEXP coefs, SEXP sigma,
    SEXP dpar, SEXP gpar, int num, SEXP fixed) {

int i = 0, j = 0;
double *beta = REAL(coefs), *sd = REAL(sigma), *gen = NULL;
SEXP generated;

  /* get the column for the generated observations. */
  generated = VECTOR_ELT(result, cur);
  gen = REAL(generated);

  if (fixed != R_NilValue) {

    rbn_gaussian_fixed(fixed, gen, num);

  }/*THEN*/
  else {

    int *dp = INTEGER(dpar), *gp = INTEGER(gpar);
    int ndp = length(dpar), ngp = length(gpar), **dcol = NULL;
    int *nlvls = NULL, *config = NULL, config_nlvl = 0;
    double **gcol = NULL, *beta_offset = NULL;
    SEXP temp;

    /* separate discrete and continuous parents. */
    gcol = Calloc1D(ngp, sizeof(double *));
    dcol = Calloc1D(ndp, sizeof(int *));
    nlvls = Calloc1D(ndp, sizeof(int));

    for (i = 0; i < ngp; i++)
      gcol[i] = REAL(VECTOR_ELT(parents, gp[i] - 1));

    for (i = 0; i < ndp; i++) {

      temp = VECTOR_ELT(parents, dp[i] - 1);
      dcol[i] = INTEGER(temp);
      nlvls[i] = NLEVELS(temp);

    }/*FOR*/

    /* generate configurations from the discrete parents. */
    config = Calloc1D(num, sizeof(int));
    c_fast_config(dcol, num, ndp, nlvls, config, &config_nlvl, 0);

    for (i = 0; i < num; i++) {

      /* if the configuration is missing, the random observation is also a
       * missing value. */
      if (config[i] == NA_INTEGER) {

        gen[i] = NA_REAL;
        continue;

      }/*THEN*/

      /* get the right set of coefficients based on the configuration of the
       * discrete parents. */
      beta_offset = beta + (ngp + 1) * config[i];
      /* initialize with intercept and standard error. */
      gen[i] = beta_offset[0] + norm_rand() * sd[config[i]];

      for (j = 0; j < ngp; j++)
        gen[i] += gcol[j][i] * beta_offset[j + 1];

    }/*FOR*/

    Free1D(gcol);
    Free1D(dcol);
    Free1D(nlvls);
    Free1D(config);

  }/*ELSE*/

}/*RBN_MIXEDCG*/

