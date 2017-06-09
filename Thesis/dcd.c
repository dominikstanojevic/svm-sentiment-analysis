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