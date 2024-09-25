import java.util.ArrayList;
import java.util.List;

public class DataSet {
    private List<double[]> data;

    public DataSet() {
        data = new ArrayList<>();
        // Dataset de entrada
        data.add(new double[]{108, 95});
        data.add(new double[]{115, 96});
        data.add(new double[]{106, 95});
        data.add(new double[]{97, 97});
        data.add(new double[]{95, 93});
        data.add(new double[]{91, 94});
        data.add(new double[]{97, 95});
        data.add(new double[]{83, 93});
        data.add(new double[]{83, 92});
        data.add(new double[]{78, 86});
        data.add(new double[]{54, 73});
        data.add(new double[]{67, 80});
        data.add(new double[]{56, 65});
        data.add(new double[]{53, 69});
        data.add(new double[]{61, 77});
        data.add(new double[]{115, 96});
        data.add(new double[]{81, 87});
        data.add(new double[]{78, 89});
        data.add(new double[]{30, 60});
        data.add(new double[]{45, 63});
        data.add(new double[]{99, 95});
        data.add(new double[]{32, 61});
        data.add(new double[]{25, 55});
        data.add(new double[]{28, 56});
        data.add(new double[]{90, 94});
        data.add(new double[]{89, 93});
    }

    public List<double[]> getData() {
        return data;
    }

    public int size() {
        return data.size();
    }
}
