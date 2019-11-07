package perceptron;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;
import java.util.Random;

public class OnlinePerceptron  {
        public static final int DIM = 3; // dimension de l'espace de representation
        public static float[] w = new float[DIM]; // parametres du modèle
        public static float[][] data = { // les observations
          {1,0,0}, {1,0,1} , {1,1,0},
          {1,1,1}
        };
    public static int[] refs = {-1, -1, -1, 1}; // les references
    public static Integer m_maxEpoch = 5;
    public static float[][] m_valData;
    public static int[] m_valRefs;
    public OnlinePerceptron( float[] W, float[][] trainData, int[] refs2,int maxEpoch) {
    	w = W;
    	data = trainData; 
    	refs = refs2;
    	m_maxEpoch = maxEpoch;
	}

	/**
	 * initialise le perceptron
	 * @param W
	 * @param trainData
	 * @param refs2
	 * @param maxEpoch
	 * @param valData
	 * @param valRefs
	 */
    public OnlinePerceptron( float[] W, float[][] trainData, int[] refs2,int maxEpoch, float[][] valData, int [] valRefs) {
    	w = W;
    	data = trainData; 
    	refs = refs2;
    	m_maxEpoch = maxEpoch;
    	m_valData = valData;
    	m_valRefs = valRefs;
	}
	/* affiche des poids.
     * @param w poids a afficher.
     * 
     * @result affichage dans la console.
     */
    public void printW() {
    	for (int i = 0; i < w.length; i += 1){
    		System.out.println("w" + i + " = " + w[i]);
    	}
    }
    /* Affiche lequation de la separatrice definie par w.
     * @param w : array de poids
     */
    public static String genWEquation() {
    	if (w[2] != 0) {
    		return ( w[1]/-w[2] + " * x + (" + w[0]/-w[2] + ')');
    	}
    	else {
    		return ""+ ( -w[0]/w[1] );
    	}
    	
    }
    /*donne la somme que le perceptron calcule.
     * @param point dont la somme est calucle
     * @result somme du point.
     */
    public static float computeSum(float[] point) {
    	float sum = 0;
    	for (int i =0; i < point.length; i += 1) {
    		sum +=  w[i] * point[i];
    	}
    	return sum;
    }

    /*donne la classe que le perceptron calcule.
     * @param point dont la classe est calucle
     * @result 1 ou -1 la classe du point.
     */
    public static int computeClass(float[] point){
    	if (computeSum(point) == 0) return 0;
    	return computeSum(point) > 0 ? 1 : -1;
    }
    
    /*ajuste les poids  de w.
     * @param eta taux d'apprentissage.
     * @param float[] point coordonnees du point a partir desquelles w est ajuste.
     * @param pointClass donne la direction dans laquelle descendre le gradient.
     * @result nouvelle valeur de w.
     */
    public static void learnFromPoint(float eta, float [] point, int pointClass) {
    	for (int i = 0; i < point.length; i += 1) {
    		w[i] += eta * point[i] * pointClass;
    	}
    }
    
    
    /* fait evoluer w pour une epoque
     * @param eta le taux d'apprentissage
     * @param data,refs les pairs points, etiquettes de A.
     * @param w initial
     * 
     * @result nouveau w.
     * @result nombre d'erreurs
     */
    public static int stage(float eta) {
    	int nbErreurs =0;
    	for (int i = 0; i < data.length; i +=1) {
    		if (computeClass(data[i]) != refs[i]) {
    			learnFromPoint(eta, data[i], refs[i]);
    			
    			nbErreurs += 1;
    		}
    	}
    	return nbErreurs;
    }

    /* initialise aleatoirement les poids wi
     * @result les poids initialises.
     */
    private static void initWeightsRandomly(float weights[]) {
    	Random randomGenerator = new Random();
    	for (int i = 0; i < w.length; i += 1) {
    		w[i] = randomGenerator.nextInt(6) - 3;
    	}
    }
    
    
    /* Trouve les poids grace a l algo du perceptron
     * @param data coordonnees 
     * @param maxStage nombre maximum d epoques.
     * @param w les poids initiaux.
     * 
     * @result le perceptron final.
     */
    public int learn() {
    	int i;
    	for (i = 0; i < m_maxEpoch; i += 1) {
    		if (stage(m_maxEpoch/(i+10.f)) == 0) break;
    	}
    	return i;
    }

	/**
	 * calcule le nombre d'erreur du perceptron avec son jeu de donnée
	 * @return un int, le nombre d'erreur
	 */
	public int nbErreursVal() {
		int nbErreurs =0;
    	for (int i = 0; i < m_valData.length; i +=1) {
    		if (computeClass(m_valData[i]) != m_valRefs[i]) {
    			nbErreurs += 1;
    		}
    	}
    	return nbErreurs;
	}

	/**
	 * calcule le nombre d'erreur sur le jeu de donnée d'entrainement
	 * @return un int, le nombre d'erreur
	 */
	public int nbErreursTrain() {
		int nbErreurs =0;
    	for (int i = 0; i < data.length; i +=1) {
    		if (computeClass(data[i]) != refs[i]) {
    			nbErreurs += 1;
    		}
    	}
    	return nbErreurs;
	}
	
    public int[] stageWithValidation(float eta) {
    	for (int i = 0; i < data.length; i +=1) {
    		if (computeClass(data[i]) != refs[i]) {
    			learnFromPoint(eta, data[i], refs[i]);
    		}
    	}
    	int [] errors = new int[2];
    	errors[0] = nbErreursTrain();
    	errors[1] = nbErreursVal();
    	return errors;
    }
    
    public int[][] learnWithValidationCurve(float eta) {
    	int i;
    	int [] errorsTemp = new int[2];
    	int[][] errorCurvesCoords = new int[m_maxEpoch][2];
    	int lastError;
    	for (i = 0; i < m_maxEpoch; i += 1) {
    		errorsTemp = stageWithValidation(eta);
    		errorCurvesCoords[i] = errorsTemp;
    		if(errorsTemp[0] == 0) {
    			break;
    		}
    	}
    	if (i < m_maxEpoch) {
        	lastError = errorCurvesCoords[i][1];
        	for (i = i; i < m_maxEpoch; i += 1) {
        		errorCurvesCoords[i][1] = lastError;
        	}
    	}
    	return errorCurvesCoords;
    }
    
    
    private static void genGnuPlot(String name) {
    	try {
			PrintWriter writer = new PrintWriter(name + ".gnu" , "UTF-8");
	        writer.println("set terminal png size 1080,1080 enhanced font 'Verdana,10'\n" + 
	        		"\n" + 
	        		"set yrange [-2:2] \n"+
	        		"set output \"curve.png\"\n" + 
	        		"set style line 1 linecolor rgb '#0060ad' linetype 1 linewidth 2\n");
	        
	        for (int i =0; i < data.length; i += 1) {
	        	writer.println("set obj "+ (i+1) +
	        			" circle at " +(data[i][1]) +
	        			","+
	        			(data[i][2]) +
	        			" fc rgb " +
	        			((refs[i]>0)?"\"red\"" : "\"blue\"")+
	        			" size 0.1");
	        }
	        writer.println("plot [-2:2] " +
	        		genWEquation());
	        System.out.println("senk");
	        writer.close();
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			System.out.println(40);
			e.printStackTrace();
		}
    }

}