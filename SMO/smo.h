#ifndef SMO_LIBRARY_H
#define SMO_LIBRARY_H

typedef struct DecisionFunction {
    double *alpha;
    double b;
} DecisionFunction;

typedef struct Parameters {
    double *input;
    int *response;
    int examples;
    int features;
    double C;
    double (*ker_func)(double *, double *, int);
    double gamma;
    double tol;
} Parameters;

DecisionFunction *smo(Parameters *);
double rbf_kernel(double *, double *, int);
#endif