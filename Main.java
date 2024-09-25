import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        DataSet dataset = new DataSet();
        List<double[]> data = dataset.getData();
        int totalSize = dataset.size();

        // Variables para guardar el mejor modelo y sus parámetros
        double bestR2 = Double.NEGATIVE_INFINITY;
        double[] bestCoefficients = new double[3];
        QuadraticRegression bestQR = null;

        // Realizar el proceso de segmentación 2 veces
        for (int run = 1; run <= 2; run++) {
            // Mezclar los datos aleatoriamente
            Collections.shuffle(data, new Random());

            // 70% para entrenamiento y 30% para pruebas
            int trainSize = (int) (0.7 * totalSize);
            double[][] X_train = new double[trainSize][1];
            double[] y_train = new double[trainSize];

            for (int i = 0; i < trainSize; i++) {
                X_train[i][0] = data.get(i)[0]; // Batch Size
                y_train[i] = data.get(i)[1];     // Variable dependiente
            }

            QuadraticRegression qr = new QuadraticRegression();  // Crear nueva instancia en cada ejecución
            qr.fit(X_train, y_train);
            
            // Hacer predicciones en el conjunto de prueba
            double[][] X_test = new double[totalSize - trainSize][1];
            double[] y_test = new double[totalSize - trainSize];
            double[] y_pred = new double[totalSize - trainSize];

            for (int i = trainSize; i < totalSize; i++) {
                X_test[i - trainSize][0] = data.get(i)[0];
                y_test[i - trainSize] = data.get(i)[1];
                y_pred[i - trainSize] = qr.predict(X_test[i - trainSize][0]);
            }

            // Calcular R²
            double r2 = qr.calculateR2(y_test, y_pred);
            double mse = qr.calculateMSE(y_test, y_pred);
            double rmse = qr.calculateRMSE(mse);
            double correlation = qr.calculateCorrelation(y_test, y_pred);

            /// Imprimir resultados de cada proceso
            System.out.println("-------Proceso " + run + " (Segmentacion 70%-30%)---------");
            System.out.printf("B0: %.15f, B1: %.15f, B2: %.15f%n", qr.getB0(), qr.getB1(), qr.getB2());
            System.out.println("Correlacion (R): " + correlation);
            System.out.println("Coeficiente de Error (MSE): " + mse);
            System.out.println("Raiz del Error Cuadratico Medio (RMSE): " + rmse); // Imprime el RMSE
            System.out.println("Coeficiente de Determinacion (R^2): " + r2);


            // Guardar los coeficientes del mejor modelo
            if (r2 > bestR2) {
                bestR2 = r2;
                bestCoefficients[0] = qr.getB0();
                bestCoefficients[1] = qr.getB1();
                bestCoefficients[2] = qr.getB2();
                bestQR = qr;  // Guardar la mejor instancia de QuadraticRegression
            }
        }

        if (bestQR != null) {
            // Imprimir los mejores coeficientes
            System.out.println("\n--------Simulaciones--------");
            System.out.println("Ecuacion:");
            System.out.printf("Y = %.15f + %.15f*(x) + %.15f*(x^2)%n", bestCoefficients[0], bestCoefficients[1], bestCoefficients[2]);

            // Realizar predicciones hardcoded
            double[] testValues = {100, 221, 98, 115, 130}; // Valores conocidos y desconocidos
            for (double value : testValues) {
                double prediction = bestQR.predict(value);
                System.out.printf("X = %.0f , Y = %.6f%n", value, prediction);
            }
        } else {
            System.out.println("No se encontro un modelo válido.");
        }
    }
}

