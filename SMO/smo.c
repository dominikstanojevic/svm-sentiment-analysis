#include "smo.h"
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

static double *alpha;
static double *w;
static double *x;
static int *y;
static int examples;
static int features;
static double eps;
static double C;
static double *Q;
static int *indexes;
static int nActive;
static int maxIters;

int isClose(double, double);

double dot(double *, double *, int);

double getProjectedGradient(double gradient, int);

void initQ(double);

void updateWeights(double oldAlpha, int index);

DecisionFunction *smo(Parameters *params) {
    srand((unsigned int) time(NULL));
    x = params->input;
    y = params->response;
    examples = params->examples;
    features = params->features;
    C = params->C;
	eps = params->tol;
	maxIters = params->iter;

    double U = INFINITY;
    double D = C * 0.5;

    //init Q
    initQ(D);

    indexes = (int *) malloc(examples * sizeof(int));
    for (int i = 0; i < examples; i++) {
        indexes[i] = i;
    }

    //init alpha
    alpha = (double *) malloc(examples * sizeof(double));
    assert(alpha != NULL);
    for (int i = 0; i < examples; i++) alpha[i] = 0;

    //init weights
    w = (double *) malloc(features * sizeof(double));
    assert(w != NULL);
    for (int i = 0; i < features; i++) w[i] = 0;

    nActive = examples;

    double Mbar = INFINITY, mbar = -INFINITY;

    int iter = 0;
    while (iter < maxIters) {
        double M = -INFINITY, m = INFINITY;

        //permutation
        for (int i = 0, swaps = nActive / 2; i < swaps; i++) {
            int first = rand() % nActive;
            int second = rand() % nActive;

            int temp = indexes[first];
            indexes[first] = indexes[second];
            indexes[second] = temp;
        }

        for (int i = 0; i < nActive; i++) {
            int index = indexes[i];


            double gradient = y[index] * dot(w, x + index * features, features) - 1 + D * alpha[index];
            double PG = 0;
            if (alpha[index] == 0) {
                if (gradient > Mbar) {
                    nActive--;
                    int temp = indexes[i];
                    indexes[i] = indexes[nActive];
                    indexes[nActive] = temp;
                    i--;
                    continue;
                } else if (gradient < 0) {
                    PG = gradient;
                }
            } else if (alpha[index] == U) {
                if (gradient < mbar) {
                    nActive--;
                    int temp = indexes[i];
                    indexes[i] = indexes[nActive];
                    indexes[nActive] = temp;
                    i--;
                    continue;
                } else if (gradient > 0) {
                    PG = gradient;
                }
            } else {
                PG = gradient;
            }

            M = M > PG ? M : PG;
            m = m < PG ? m : PG;

            if (fabs(PG) > 1.0e-12) {
                double oldAlpha = alpha[index];
                double a = alpha[index] - gradient / Q[index];
                if (a < 0) a = 0;
                else if (a > U) a = U;
                alpha[index] = a;

                updateWeights(oldAlpha, index);
            }
        }

        iter++;
      
        if (M - m <= eps) {
            if (nActive == examples) {
                break;
            } else {
                nActive = examples;
                Mbar = INFINITY;
                mbar = -INFINITY;
                continue;
            }
        }

        if (M <= 0) Mbar = INFINITY;
        else Mbar = M;

        if (m >= 0) mbar = -INFINITY;
        else mbar = m;
    }

    DecisionFunction *function = (DecisionFunction *) malloc(sizeof(DecisionFunction));
    function->weights = w + 1;
	function->bias = *w;
	
	free(alpha);
	free(Q);
	free(indexes);
	
    return function;
}


void updateWeights(double oldAlpha, int index) {
    double scalar = (alpha[index] - oldAlpha) * y[index];
    double *xIndex = x + index * features;
    for (int i = 0; i < features; i++) {
        w[i] += scalar * xIndex[i];
    }
}

void initQ(double D) {
    Q = (double *) malloc(examples * sizeof(double));

    for (int i = 0; i < examples; i++) {
        double *offset = x + i * features;
        Q[i] = dot(offset, offset, features) + D;
    }
}

double dot(double *first, double *second, int len) {
    double sum = 0;
    for (int i = 0; i < len; i++) {
        sum += first[i] * second[i];
    }

    return sum;
}

/*int main(void) {
    Parameters params;
    params.examples = 50000;
    params.features = 1001;

    int u = params.examples * params.features;

    srand((unsigned int) time(NULL));
    double *x = (double *)malloc(u * sizeof(double));
    int y[params.examples];

    for (int i = 0; i < u; i++) {
        if(i % params.features == 0) {
            x[i] = 1;
        } else {
            x[i] = ((double)rand()) / (rand() + 1);
        }

        if (i < params.examples) {
            y[i] = rand() % 2 ? -1 : 1;
        }
    }

    params.input = x;
    params.response = y;
    //params.ker_func = rbf_kernel;
    params.gamma = 1. / params.features;
    params.tol = 1e-4;
    params.C = 0.25;

    DecisionFunction *function = smo(&params);

    int total = 0;
    for(int i = 0; i < params.examples; i++) {
        double *ex = x + i * features;
        int pred;
        double c = dot(ex, function->alpha, features);
        if (c < 0) {
            pred = -1;
        } else {
            pred = 1;
        }

        if (pred == y[i]) {
            total++;
        }
    }

    printf("%d\n", total);

    free(function->alpha);
    free(function);
}*/
