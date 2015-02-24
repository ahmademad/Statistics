
public class Logistic {

    private double sigmoid(double x) {
    	double S = 1/(1+Math.exp(x));
    	return S;
    }

   public static void main(String [] args) {
   	Logistic o =  new Logistic();
   	System.out.println(o.sigmoid(5));
   }
}
