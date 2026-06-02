/**
 * 
 */
package ceka.utils;

/**
 * some stochastic function
 *
 */
public class Stochastics {
	/**
	 * fact n!
	 * @param n
	 * @return
	 */
	public static double fact(long n)
	{
		double x = 1;
		for (int i = 1; i <= n; i++)
			x *= i;
		return x;
	}
	
	/**
	 * combination m! / (n!(m-n)!)
	 * @param m
	 * @param n
	 * @return
	 */
	public static double combination(int m, int n)
	{
		//m!/n!(m-n)!
		return fact(m) / (fact(n) * fact(m-n));
	}
	
	/**
	 * use binomial model to integrate labels
	 * @param nW number of workers
	 * @param p probability to provide right answer
	 * @return
	 */
	public static double binomialIntegration(int nW, double p) {
		int N = nW/2;
		double r = 0.0;
		for (int i =0; i <= N; i++)
			r += (combination(2 * N + 1, i)*Math.pow(p, 2 * N + 1 - i)* Math.pow(1-p, i));
		return r;
	}

}
