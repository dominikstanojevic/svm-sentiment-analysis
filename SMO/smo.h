#ifndef SMO_LIBRARY_H
#define SMO_LIBRARY_H

typedef struct DecisionFunction {
    double *weights;
    double bias;
} DecisionFunction;

typedef struct Parameters {
    double *input;
    int *response;
    int examples;
    int features;
    double C;
    double tol;
	double iter;
} Parameters;

DecisionFunction *smo(Parameters *);
#endif