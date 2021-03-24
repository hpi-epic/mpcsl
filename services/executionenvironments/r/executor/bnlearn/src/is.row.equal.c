#include "include/rcore.h"

#define ARC(i,col) CHAR(STRING_ELT(data, i + col * nrow))

/* which elements of "data" match "array"? */
SEXP is_row_equal(SEXP data, SEXP array) {

int i = 0, nrow = length(data) / 2;
const char *from = CHAR(STRING_ELT(array, 0));
const char *to = CHAR(STRING_ELT(array, 1));
SEXP res;

  /* allocate the return value, which is an array holding a
   * logical value for each arc in the arc set. */
  PROTECT(res = allocVector(LGLSXP, nrow));

  for (i = 0; i < nrow; i++) {

    /* check the first element; if it does no match skip the second one. */
    if (!strcmp(from, ARC(i, 0)) ) {

      /* if the first element matches, check the other one. */
      if (!strcmp(to, ARC(i, 1)) ) {

        LOGICAL(res)[i] = TRUE;

      }/*THEN*/
      else {

        LOGICAL(res)[i] = FALSE;

      }/*ELSE*/

    }/*THEN*/
    else {

      LOGICAL(res)[i] = FALSE;

    }/*ELSE*/

  }/*FOR*/

  UNPROTECT(1);

  return res;

}/*IS_ROW_EQUAL*/

#undef ARC
#define ARC(i,col) CHAR(STRING_ELT(set, i + col * nrow))

/* does "arc" match any elements of "set"? */
SEXP is_listed(SEXP arc, SEXP set, SEXP either, SEXP both, SEXP debug) {

int i = 0, matched = 0, nrow = length(set) / 2;
const char *from = CHAR(STRING_ELT(arc, 0));
const char *to = CHAR(STRING_ELT(arc, 1));
bool debugging = isTRUE(debug);

  /* if the arc set is NULL, return immediately. */
  if (isNull(set))
    return ScalarLogical(FALSE);

  for (i = 0; i < nrow; i++) {

    if (debugging)
      Rprintf("* checking %s -> %s\n", ARC(i, 0), ARC(i, 1));

    /* check the first element; if it does not match skip to the second one. */
    if (!strcmp(from, ARC(i, 0)) ) {

      /* if the second element matches, return if "both = FALSE" or if
       * "both = TRUE" and the reversed arc has been already found out. */
      if (!strcmp(to, ARC(i, 1)) ) {

        /* increment the "matched" counter, which is needed to be sure both
         * A -> B and B -> A are in the arc set when "both = TRUE". */
        matched++;

        if (debugging)
          Rprintf("  > matched %s -> %s (matched is %d).\n", ARC(i, 0),
            ARC(i, 1), matched);

        /* return TRUE if one of the following conditions is met:
         *
         * 1) exact match (either = both = FALSE).
         * 2) match regardless of direction (either = TRUE).
         * 3) match both directions (both = TRUE) when the other
         *      one has already been found (matched = 2).
         */
        if ((!isTRUE(either) && !isTRUE(both)) || isTRUE(either) ||
             ((matched == 2) && isTRUE(both)))
             goto success;

      }/*THEN*/

    }/*THEN*/
    else if (isTRUE(either) || isTRUE(both)) {

      /* the same as above, but with the reversed arc; this part is
       * skipped if "both = FALSE" and "either = FALSE", since the
       * reversed arc should not be matched in that case. */

      if (!strcmp(to, ARC(i, 0)) ) {

        if (!strcmp(from, ARC(i, 1)) ) {

          /* increment the "matched" counter, which is needed to be sure both
           * A -> B and B -> A are in the arc set when "both = TRUE". */
          matched++;

          if (debugging)
            Rprintf("  > matched %s -> %s (matched is %d).\n", ARC(i, 0),
              ARC(i, 1), matched);

          /* return TRUE if one of the following conditions is met:
           *
           * 1) match regardless of direction (either = TRUE).
           * 2) match both directions (both = TRUE) when the other
           *      one has already been found (matched = 2).
           */
          if (isTRUE(either) || ((matched == 2) && isTRUE(both)))
            goto success;

        }/*THEN*/

      }/*THEN */

    }/*THEN*/

  }/*FOR*/

  return ScalarLogical(FALSE);

success:
   return ScalarLogical(TRUE);

}/*IS_LISTED*/

