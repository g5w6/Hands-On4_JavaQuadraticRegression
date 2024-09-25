public class QuadraticRegression {
    private double B0, B1, B2;

    public void fit(double[][] X, double[] y) {
        int n = X.length;

        // Inicializar los sumatorios necesarios para la matriz de Gauss-Jordan
        double sumX = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0;
        double sumY = 0, sumXY = 0, sumX2Y = 0;

        for (int i = 0; i < n; i++) {
            double xi = X[i][0];
            double yi = y[i];

            sumX += xi;
            sumX2 += xi * xi;
            sumX3 += xi * xi * xi;
            sumX4 += xi * xi * xi * xi;
            sumY += yi;
            sumXY += xi * yi;
            sumX2Y += xi * xi * yi;
        }

        // Matriz aumentada de 3x4 para el sistema de ecuaciones lineales
        double[][] matrix = {
            {n, sumX, sumX2, sumY},
            {sumX, sumX2, sumX3, sumXY},
            {sumX2, sumX3, sumX4, sumX2Y}
        };

        // Resolver la matriz mediante Gauss-Jordan
        gaussJordan(matrix);

        // Extraer los coeficientes
        B0 = matrix[0][3];
        B1 = matrix[1][3];
        B2 = matrix[2][3];
    }

    // Método para aplicar eliminación de Gauss-Jordan
    private void gaussJordan(double[][] matrix) {
        int n = matrix.length;

        for (int i = 0; i < n; i++) {
            // Hacer que el elemento en matrix[i][i] sea 1
            double divisor = matrix[i][i];
            for (int j = 0; j < n + 1; j++) {
                matrix[i][j] /= divisor;
            }

            // Hacer que todos los elementos en la columna i (excepto el pivote) sean 0
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = matrix[k][i];
                    for (int j = 0; j < n + 1; j++) {
                        matrix[k][j] -= factor * matrix[i][j];
                    }
                }
            }
        }
    }

    // Predicción usando los coeficientes obtenidos
    public double predict(double x) {
        return B0 + B1 * x + B2 * x * x;
    }

    // Obtener los coeficientes
    public double getB0() {
        return B0;
    }

    public double getB1() {
        return B1;
    }

    public double getB2() {
        return B2;
    }

    // Cálculo del R^2 (coeficiente de determinación)
    public double calculateR2(double[] yTrue, double[] yPred) {
        double ssRes = 0, ssTot = 0;
        double meanY = 0;

        for (double v : yTrue) {
            meanY += v;
        }
        meanY /= yTrue.length;

        for (int i = 0; i < yTrue.length; i++) {
            ssRes += Math.pow(yTrue[i] - yPred[i], 2);
            ssTot += Math.pow(yTrue[i] - meanY, 2);
        }

        return 1 - (ssRes / ssTot);
    }

    // Cálculo del MSE
    public double calculateMSE(double[] yTrue, double[] yPred) {
        double mse = 0;
        for (int i = 0; i < yTrue.length; i++) {
            mse += Math.pow(yTrue[i] - yPred[i], 2);
        }
        return mse / yTrue.length;
    }

    // Cálculo del RMSE
    public double calculateRMSE(double mse) {
        return Math.sqrt(mse);
    }

    public double calculateCorrelation(double[] y_test, double[] y_pred) {
        int n = y_test.length;
        double sumYTest = 0, sumYPred = 0, sumYTestYPred = 0;
        double sumYTestSquared = 0, sumYPredSquared = 0;
    
        for (int i = 0; i < n; i++) {
            sumYTest += y_test[i];
            sumYPred += y_pred[i];
            sumYTestYPred += y_test[i] * y_pred[i];
            sumYTestSquared += y_test[i] * y_test[i];
            sumYPredSquared += y_pred[i] * y_pred[i];
        }
    
        double numerator = n * sumYTestYPred - sumYTest * sumYPred;
        double denominator = Math.sqrt((n * sumYTestSquared - sumYTest * sumYTest) * (n * sumYPredSquared - sumYPred * sumYPred));
    
        if (denominator == 0) {
            return 0; // Evita la división por cero
        } else {
            return numerator / denominator;
        }
    }
    
}