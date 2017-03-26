#include "smo.h"
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>


static double **Q;
static double *G;
static int examples;
static int *y;
static double C;
static double tol;
static double tau = 1e-12;
static double gamma;

static double *alpha;

void setParams(Parameters *pParameters);

void initQ(Parameters *params);

void initGradient();

void initAlpha();

void selectWorkingSet(int *i, int *j);

double calculateBias();

DecisionFunction *smo(Parameters *parameters) {
    setParams(parameters);
    initQ(parameters);
    initGradient();
    initAlpha();

    int i, j;
    while (1) {
        selectWorkingSet(&i, &j);
        if (j == -1) {
            break;
        }

        double a = Q[i][i] + Q[j][j] - 2 * y[i] * y[j] * Q[i][j];
        if (a <= 0) {
            a = tau;
        }

        double b = -y[i] * G[i] + y[j] * G[j];

        double oldAi = alpha[i], oldAj = alpha[j];
        alpha[i] += y[i] * b / a;
        alpha[j] -= y[j] * b / a;

        double sum = y[i] * oldAi + y[j] * oldAj;
        if (alpha[i] > C) alpha[i] = C;
        else if (alpha[i] < 0) alpha[i] = 0;
        alpha[j] = y[j] * (sum - y[i] * alpha[i]);

        if (alpha[j] > C) alpha[j] = C;
        else if (alpha[j] < 0) alpha[j] = 0;
        alpha[i] = y[i] * (sum - y[j] * alpha[j]);

        double deltaAi = alpha[i] - oldAi, deltaAj = alpha[j] - oldAj;
        for (int x = 0; x < examples; x++) {
            G[x] += Q[x][i] * deltaAi + Q[x][j] * deltaAj;
        }
    }

    calculateBias();


    DecisionFunction *function = (DecisionFunction *) malloc(sizeof(DecisionFunction));
    assert(function != NULL);
    function->alpha = alpha;
    function->b = calculateBias();

    for(int x = 0; x < examples; x++) {
        free(Q[x]);
    }
    free(Q);
    free(G);

    return function;
}

double calculateBias() {
    double b;
    int free = 0;
    double ub = INFINITY, lb = -INFINITY, sum = 0;
    for(int i = 0; i < examples; i++) {
        double yG = y[i] * G[i];
        if(alpha[i] == C) {
            if (y[i] == -1) {
                ub = ub < yG ? ub : yG;
            } else {
                lb = lb > yG ? lb : yG;
            }
        } else if (alpha[i] == 0) {
            if (y[i] == 1) {
                ub = ub < yG ? ub : yG;
            } else {
                lb = lb > yG ? lb : yG;
            }
        } else {
            ++free;
            sum += yG;
        }
    }

    if (free > 0) {
        b = sum / free;
    } else {
        b = (ub + lb) / 2;
    }

    return b;
}

void selectWorkingSet(int *i, int *j) {
    *i = -1;
    double G_max = -INFINITY, G_min = INFINITY;
    for (int x = 0; x < examples; x++) {
        if ((y[x] == 1 && alpha[x] < C) || (y[x] == -1 && alpha[x] > 0)) {
            if (-y[x] * G[x] >= G_max) {
                *i = x;
                G_max = -y[x] * G[x];
            }
        }
    }

    *j = -1;
    double obj_min = INFINITY;
    for (int x = 0; x < examples; x++) {
        if ((y[x] == 1 && alpha[x] > 0) || (y[x] == -1 && alpha[x] < C)) {
            double b = G_max + y[x] * G[x];

            if (-y[x] * G[x] <= G_min) {
                G_min = -y[x] * G[x];
            }
            if (b > 0) {
                double a = Q[*i][*j] + Q[x][x] - 2 * y[*i] * y[x] * Q[*i][x];
                if(a <= 0) {
                    a = tau;
                }
                if (- (b * b) / a <= obj_min) {
                    *j = x;
                    obj_min = - (b * b) / a;
                }
            }
        }
    }

    if (G_max - G_min < tol) {
        *i = -1;
        *j = -1;
    }
}

void initAlpha() {
    alpha = (double *) malloc(examples * sizeof(double));
    assert(alpha != NULL);
    for (int i = 0; i < examples; i++) {
        alpha[i] = 0;
    }
}

void initGradient() {
    G = (double *) malloc(examples * sizeof(double));
    assert(G != NULL);
    for (int i = 0; i < examples; i++) {
        G[i] = -1;
    }
}

void initQ(Parameters *params) {
    Q = (double **) malloc(examples * sizeof(double *));
    assert(Q != NULL);

    double **rows = (double **) malloc(examples * sizeof(double *));
    assert(rows != NULL);
    for (int i = 0; i < examples; i++) {
        //get rows
        rows[i] = params->input + i * params->features;

        //init Q
        Q[i] = (double *) malloc(examples * sizeof(double));
        printf("%d\n", i);
        assert(Q[i] != NULL);
    }

    for (int i = 0; i < examples; i++) {
        for (int j = i; j < examples; j++) {
            Q[i][j] = Q[j][i] = y[i] * y[j] * params->ker_func(rows[i], rows[j], params->features);
        }
    }

    free(rows);
}

void setParams(Parameters *params) {
    examples = params->examples;
    y = params->response;
    C = params->C;
    tol = params->tol;
    gamma = params->gamma;
}

double rbf_kernel(double *first, double *second, int size) {
    double result = 0;
    for (int i = 0; i < size; i++) {
        double diff = first[i] - second[i];
        result += diff * diff;
    }

    return exp(-gamma * result);
}

int main(void) {
    double *x = (double *)malloc(50000 * 100 * sizeof(double));
    int y[50000];

    for (int i = 0; i < 5000000; i++) {
        x[i] = ((double)rand()) / (rand() + 1);

        if (i < 50000) {
            y[i] = rand() % 2 ? -1 : 1;
        }
    }

    Parameters params;
    params.input = x;
    params.response = y;
    params.ker_func = rbf_kernel;
    params.examples = 50000;
    params.features = 100;
    params.gamma = 1. / params.features;
    params.tol = 1e-3;
    params.C = 1;

    DecisionFunction *function = smo(&params);
    for (int i = 0; i < params.examples; i++) {
        printf("%f ", function->alpha[i]);
    }

    free(function->alpha);
    free(function);
}