library(RCIT)

kcitWrapper <- function(x,y,z, suffStat) {
	if (identical(z, integer(0))) {
		U_KCI(as.vector(t(suffStat$dm[x])),as.vector(t(suffStat$dm[y])))
	} else {
		KCIT(suffStat$dm[x],suffStat$dm[y],suffStat$dm[z])
	}
}