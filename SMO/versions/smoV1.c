#include "smo.h"
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

static void initFunction();

static void initKernel();

static int examineExample(int index);

static double getError(int index, int response);

static int countNonBounds();

static int secondHeuristics(double error);

static int takeStep(int i1, int i2, double alph2, int y2, double e2);

static void initBoundFlag();

static void calculateBounds(int y1, int y2, double alph1, double alph2, double *l, double *h);

static void update_b(int i1, double alph1, double e1, double a1, int i2, double alph2, double e2, double a2);

static void update_cache(int i1, double a1, int i2, double a2, double old);

static double calcOutput(int index);

static double *initArr(int size, int init);

static int isOnBounds(double value, double lower, double upper);

static int isClose(double a, double b);

static double abs_d(double number);

static Parameters *parameters = NULL;
static double *cache = NULL;
static double *tot_kernel = NULL;
static DecisionFunction *decisionFunction = NULL;
static int *bound_flag = NULL;

DecisionFunction *smo(Parameters *params) {
	parameters = params;
	initFunction();
	cache = initArr(params->examples, 1);
	initKernel();
	initBoundFlag();
	int numChanged = 0, examineAll = 1;
	while (numChanged > 0 || examineAll) {
		numChanged = 0;
		if (examineAll) {
			for (int i = 0; i < params->examples; i++) {
				numChanged += examineExample(i);
			}
		}
		else {
			for (int i = 0; i < params->examples; i++) {
				if (!bound_flag[i]) {
					numChanged += examineExample(i);
				}
			}
		}

		if (examineAll == 1) {
			examineAll = 0;
		}
		else if (numChanged == 0) {
			examineAll = 1;
		}

	}

	//fighting against memory leaks
	free(cache);
	free(tot_kernel);
	free(bound_flag);


	return decisionFunction;
}

static void initBoundFlag() {
	bound_flag = (int *)malloc(parameters->examples * sizeof(int));
	for (int i = 0; i < parameters->examples; i++) {
		bound_flag[i] = 1;
	}
}

static int examineExample(int index) {
	int y2 = parameters->response[index];
	double alph2 = decisionFunction->alpha[index];
	double e2 = getError(index, y2);
	double r2 = e2 * y2;
	if ((r2 < -parameters->tol && alph2 < parameters->C) || (r2 > parameters->tol && alph2 > 0)) {
		int index1;
		if (countNonBounds() > 1) {
			index1 = secondHeuristics(e2);
			if (takeStep(index1, index, alph2, y2, e2)) {
				return 1;
			}
		}


		int random = rand() % parameters->examples;
		for (int i = 0; i < parameters->examples; i++) {
			index1 = (random + i) % parameters->examples;
			if (!bound_flag[index1]) {
				if (takeStep(index1, index, alph2, y2, e2)) {
					return 1;
				}
			}
		}

		random = rand() % parameters->examples;
		for (int i = 0; i < parameters->examples; i++) {
			index1 = (random + i) % parameters->examples;
			if (takeStep(index1, index, alph2, y2, e2)) {
				return 1;
			}
		}
	}

	return 0;
}

static int takeStep(int i1, int i2, double alph2, int y2, double e2) {
	if (i1 == i2) return 0;
	double alph1 = decisionFunction->alpha[i1];
	int y1 = parameters->response[i1];
	double e1 = getError(i1, y1);
	int s = y1 * y2;
	double l, h;
	calculateBounds(y1, y2, alph1, alph2, &l, &h);
	if (isClose(l, h)) return 0;

	double k11 = tot_kernel[i1 * parameters->examples + i1];
	double k12 = tot_kernel[i1 * parameters->examples + i2];
	double k22 = tot_kernel[i2 * parameters->examples + i2];
	double eta = 2 * k12 - k11 - k22;

	//only for kernels which obey Mercer's condition
	double a2 = alph2 - y2 * (e1 - e2) / eta;
	if (a2 < l) a2 = l;
	else if (a2 > h) a2 = h;

	if (isClose(a2, 0)) {
		a2 = 0;
	}
	else if (isClose(a2, parameters->C)) {
		a2 = parameters->C;
	}

	if (abs_d(a2 - alph2) < parameters->tol * (a2 + alph2 + parameters->tol)) {
		return 0;
	}

	double a1 = alph1 + s * (alph2 - a2);

	double b_old = decisionFunction->b;
	update_b(i1, alph1, e1, a1, i2, alph2, e2, a2);

	bound_flag[i1] = isOnBounds(a1, 0, parameters->C);
	bound_flag[i2] = isOnBounds(a2, 0, parameters->C);

	update_cache(i1, a1, i2, a2, b_old);

	//store multipliers
	decisionFunction->alpha[i1] = a1;
	decisionFunction->alpha[i2] = a2;


	return 1;
}


static void update_cache(int i1, double a1, int i2, double a2, double b_old) {
	double tot1 = parameters->response[i1] * (a1 - decisionFunction->alpha[i1]);
	double tot2 = parameters->response[i2] * (a2 - decisionFunction->alpha[i2]);
	double b_diff = b_old - decisionFunction->b;

	for (int i = 0; i < parameters->examples; i++) {
		if (!bound_flag[i]) {
			if (i == i1 || i == i2) {
				cache[i] = 0;
			}
			else {
				double k1i = tot_kernel[i1 * parameters->examples + i];
				double k2i = tot_kernel[i2 * parameters->examples + i];
				cache[i] += tot1 * k1i + tot2 * k2i + b_diff;
			}
		}
	}
}

static void update_b(int i1, double alph1, double e1, double a1, int i2, double alph2, double e2, double a2) {
	double tot1 = parameters->response[i1] * (a1 - alph1);
	double tot2 = parameters->response[i2] * (a2 - alph2);

	double k11 = tot_kernel[i1 * parameters->examples + i1];
	double k12 = tot_kernel[i1 * parameters->examples + i2];
	double k22 = tot_kernel[i2 * parameters->examples + i2];

	double b1 = e1 + tot1 * k11 + tot2 * k12 + decisionFunction->b;
	double b2 = e2 + tot1 * k12 + tot2 * k22 + decisionFunction->b;

	decisionFunction->b = (b1 + b2) / 2;
}

static void calculateBounds(int y1, int y2, double alph1, double alph2, double *l, double *h) {
	if (y1 != y2) {
		double diff = alph2 - alph1;
		*l = diff > 0 ? diff : 0;
		*h = parameters->C + (diff < 0 ? diff : 0);
	}
	else {
		double sum = alph1 + alph2;
		*l = sum > parameters->C ? (sum - parameters->C) : 0;
		*h = sum < parameters->C ? sum : parameters->C;
	}
}

static int secondHeuristics(double error) {
	int index = -1;
	if (error > 0) {
		for (int i = 0; i < parameters->examples; i++) {
			if (!bound_flag[i] &&
				(index == -1 || cache[index] > cache[i])) {
				index = i;
			}
		}
	}
	else {
		for (int i = 0; i < parameters->examples; i++) {
			if (!bound_flag[i] &&
				(index == -1 || cache[index] < cache[i])) {
				index = i;
			}
		}
	}

	return index;
}

static int countNonBounds() {
	int total = 0;
	for (int i = 0; i < parameters->examples; i++) {
		if (!bound_flag[i]) {
			total++;
		}
	}

	return total;
}

static double getError(int index, int response) {
	if (bound_flag[index]) {
		return calcOutput(index) - response;
	}
	else {
		return cache[index];
	}
}

static double calcOutput(int index) {
	double result = -decisionFunction->b;
	double *kernel_row = tot_kernel + index * parameters->examples;
	for (int i = 0; i < parameters->examples; i++) {
		result += decisionFunction->alpha[i] * /*parameters->response[i] **/ kernel_row[i];
	}

	return result;
}

static void initKernel() {
	int examples = parameters->examples;
	tot_kernel = initArr(examples * examples, 0);

	double **rows = (double **)malloc(examples * sizeof(double *));
	for (int i = 0; i < examples; i++) {
		rows[i] = parameters->input + i * parameters->features;
	}

	for (int i = 0; i < examples; i++) {
		for (int j = i; j < examples; j++) {
			tot_kernel[i * examples + j] = tot_kernel[j * examples + i] = parameters->ker_func(rows[i], rows[j],
				parameters->features);
		}
	}

	free(rows);
}

static void initFunction() {
	decisionFunction = (DecisionFunction *)malloc(sizeof(DecisionFunction));
	decisionFunction->alpha = initArr(parameters->examples, 1);
	decisionFunction->b = 0;
}


//kernels
double rbf_kernel(double *first, double *second, int size) {
	double result = 0;
	for (int i = 0; i < size; i++) {
		double diff = first[i] - second[i];
		result += diff * diff;
	}

	return exp(-parameters->gamma * result);
}


static double *initArr(int size, int init) {
	double *arr = (double *)malloc(size * sizeof(double));
	if (init) {
		for (int i = 0; i < size; i++) {
			arr[i] = 0;
		}
	}

	return arr;
}

static double tol = 1e-12;
static int isClose(double a, double b) {
	return abs_d(a - b) < tol;
}

static int isOnBounds(double value, double lower, double upper) {
	return isClose(value, lower) || isClose(value, upper);
}

static double abs_d(double number) {
	return number < 0 ? -number : number;
}

int main(void) {
	double *x = (double *)malloc(10000 * 100 * sizeof(double));
	int y[10000];

	for (int i = 0; i < 1000000; i++) {
		x[i] = ((double)rand()) / (rand() + 1);

		if (i < 10000) {
			y[i] = rand() % 2 ? -1 : 1;
		}
	}

	Parameters params;
	params.input = x;
	params.response = y;
	params.ker_func = rbf_kernel;
	params.examples = 10000;
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