package perceptron;

import java.util.Iterator;
import java.util.Random;
import java.util.ArrayList;

public class PerceptronMulti {
	public int m_pointDim = 12;
	public int m_stickersDim = 10;
	private float m_perceptronWeightsArray [][] = new float[m_stickersDim][m_pointDim];


	/**
	 * fonction d'initialisastion du perceptron
	 * @param pointDim
	 * @param stickersDim
	 * @param weights
	 */
	public PerceptronMulti(int pointDim,int stickersDim, float weights[][]) {
		m_pointDim = pointDim;
		m_stickersDim = stickersDim;
		m_perceptronWeightsArray = weights;
	}

	/**genRandomWeights
	 * fonction qui prend en paramtre un entier alpha et initialise les poid du perceptron de facon aleatoire mais
	 * au meme ordre de grandeur
	 * @param alpha
	 */
	private void genRandomWeights(float alpha) {
		Random randGen= new Random();
		
		for (int i = 0; i < m_stickersDim; i += 1) {
			for (int j = 0; j < m_pointDim; j += 1) {
				m_perceptronWeightsArray[i][j] = randGen.nextFloat() * alpha;
			}
		}
	}

	/**
	 * fonction d'initialisation du perceptron
	 * @param pointDim
	 * @param stickersDim
	 */
	public PerceptronMulti(int pointDim,int stickersDim) {
		m_pointDim = pointDim;
		m_stickersDim = stickersDim;
		m_perceptronWeightsArray = new float[stickersDim][pointDim];
		genRandomWeights(1.f/(pointDim));
	}

	/**
	 * fonction qui prend en parametre un entier a representant une etiquette
	 * @param a
	 * @return un tableau d'entier ou de la taille du noubre de classe remplie de 0 et ou la seule valeur non nulle
	 * est a l'indice a
	 */
	public int[] intToSticker(int a) {
		int [] sticker = new int[m_stickersDim];
		for(int i = 0; i < m_stickersDim; i ++){  //rajout de la boucle pour initialiser les autres valeurs a 0
			sticker[i] = 0;
		}
		sticker[a%m_stickersDim] = 1;
		return sticker;
	}

	/**
	 * fonction qui prend en parametre deux tableaux d'entier representant deux vecteurs
	 * @param a
	 * @param b
	 * @return un floatant representant le resultat du produit vectoriel entre a et b
	 */
	private float dotProd(float[] a, float[] b) {
		float sum = 0;
		for (int i =0; i < m_pointDim; i += 1) {
			sum += a[i] * b[i];
		}
		return sum;
	}

	/**
	 * fonction qui prend en parametre un tableau de floattant representant une donnée
	 * @param point
	 * @return un tableau de floattant representant les probabilité que la donnée a d'appartenir a chaque calsse
	 */
	public float[] probaForPoint(float[] point){
		float [] proba = new float[m_stickersDim];
		float [] expFromPointAndWeight = new float[m_stickersDim];
		float expSum = 0;
		float a = 0;
		for (int expIndex = 0; expIndex < m_stickersDim; expIndex += 1) {
			a = (float) Math.pow(Math.E,dotProd(point,m_perceptronWeightsArray[expIndex]));
			expFromPointAndWeight[expIndex] = a;
			expSum += a;
		}
		
		for (int perceptronIndex = 0; perceptronIndex < m_stickersDim; perceptronIndex += 1) {
			proba[perceptronIndex] = expFromPointAndWeight[perceptronIndex] / expSum;
		}
		return proba;
	}


	/**
	 * fonction qui prend en parametre un point(tableau de float) le tableau associé a son etiquette(tableau de int)
	 * eta le taux d'apprentissage(un int). la fonction met a jour les parametre du perceptron
	 * @param point
	 * @param sticker
	 * @param eta
	 */
	private void learnFromPoint(float point[], int [] sticker,float eta) {
		float[] probas = probaForPoint(point);
		for (int perceptronIndex =0; perceptronIndex < m_stickersDim; perceptronIndex+= 1) {
			for (int weightIndex = 0; weightIndex < m_pointDim; weightIndex+=1) {
				m_perceptronWeightsArray[perceptronIndex][weightIndex] -=
						eta * point[weightIndex] *(probas[perceptronIndex] - sticker[perceptronIndex]);
			}
		}
	}

	/**
	 * fonction qui prend en parametre un point(un tableau de float)
	 * @param point
	 * @return un entier representant la classe de la donnée estimé par le perceptron
	 */
	public int computeClass (float point[]) {
		float [] proba = probaForPoint(point);
		float maxProb = proba[0];
		int maxIndex = 0;
		for (int i =1; i < m_stickersDim; i += 1) {
			if (proba[i] > maxProb) {
				maxProb = proba[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}

	/**
	 * fonction qui prend en parametre toute les données(tableau de tableau de float), leur etiquette associés
	 * (tableau de int), les tableau issue de intToSticker de chaque donnée(tableau de tableau de int) et le taux
	 * d'apprentissage eta (un float) et fait un epoque avec le jeu de donnée
	 * @param data
	 * @param dataLabels
	 * @param dataStickers
	 * @param eta
	 */
	public void stage(float data[][], int [] dataLabels, int [][] dataStickers, float eta) {
		int errorsNb = 0;
    	for (int i = 0; i < data.length; i +=1) {
			learnFromPoint(data[i], dataStickers[i], eta);
		}
	}

	/**fonction qui prend en parametre le jeu de donnée (tableau de tableau de float), les label associé(tableau de int),
	 * le taux d'apprentissage(un float), le nombre max d'époque(un int). fait faire epoque max epoque au jeu de donnée
	 * @param data
	 * @param dataLabels
	 * @param eta
	 * @param maxStages
	 */
	public void learn(float data[][], int dataLabels[], float eta, int maxStages) {
		int [][]dataStickers = new int [dataLabels.length][m_stickersDim];
		for (int i =0; i < dataLabels.length; i += 1) {
			dataStickers[i] = intToSticker(dataLabels[i]);
		}
		for (int stageIndx = 0; stageIndx < maxStages; stageIndx += 1) {
			stage(data,dataLabels,dataStickers,eta);
		}
	}

	/**
	 * fonction qui prend en parametre le jeu de donnée(un tableau de tableau de float), les labels associés(tableau de int)
	 * @param data
	 * @param dataLabels
	 * @return un int qui represente le nombre d'erreur sur le jeux de donnée par rapport au jeux de donnée actuel
	 */
	public int errorsDataSet(float data[][], int dataLabels[]) {
		int nbErrors = 0;
		for (int i = 0; i < data.length; i += 1) {
			if (this.computeClass(data[i]) != dataLabels[i]) {
				nbErrors += 1;
			}
		}
		return nbErrors;
	}

	/**
	 * fonction qui prend en parametre le jeu de donnée(tableau de tableau de float), les labels associés(un tableau de
	 * int)
	 * @param data
	 * @param dataLabels
	 * @return la valeur de la fonction de cout lier au jeu de donnée
	 */
	private float costFunction(float data[][], int [] dataLabels) {
		float Etotal = 0;
		for (int imageIndex = 0; imageIndex < data.length; imageIndex += 1) {
 			Etotal += (float) Math.log(probaForPoint(data[imageIndex])[dataLabels[imageIndex]]);
		}
		return Etotal/data.length;
	}

	/**
	 * @return a matrix, with numbers in slots:
	 * 	 * 						which lines (first index) indexes, are the class number the perceptron gave.
	 * 	 *	 					which columns (second index) indexes, are the class number they really are.
	 * @param data points
	 * @param dataLabels points class
	 */
	public int[][] confusionMatrix(float data [][], int [] dataLabels){
		int [][] confMat = new int[m_stickersDim][m_stickersDim];
		for (int columnIndex = 0; columnIndex < m_stickersDim; columnIndex += 1){
			for (int dataIndex =0; dataIndex < dataLabels.length; dataIndex+= 1){
				if (columnIndex == dataLabels[dataIndex]){
						int lineIndex = computeClass(data[dataIndex]);
						confMat[lineIndex][columnIndex] += 1;
				}
			}
		}
		return confMat;
	}

	/**
	 *
	 * @param data
	 * @param dataLabels
	 * @return la representation de la matrice de confusion.
	 */
	public String stringConfusionMatrix(float data [][], int [] dataLabels){
		int[][] confMat = confusionMatrix(data,dataLabels);
		String ret = tabToString(confMat[0]);
		for (int i = 1; i < m_stickersDim; i+=1){
			ret += '\n'+tabToString(confMat[i]);
		}
		return ret;
	}

	/**
	 *
	 * @param data training points
	 * @param dataLabels training points's classes
	 *
	 * @param dataValidation validation's points
	 * @param dataValidationLabels validation's classes
	 *
	 * @param eta learning coefficient
	 * @param maxStages number of repetition.
	 *
	 * @return data for graph (int[maxStages][]
	 */
	public int[][] learnWithErrorsArray(float data[][], int dataLabels[],
										float dataValidation[][], int dataValidationLabels[],
										float eta, int maxStages) {
		int [][]dataStickers = new int [dataLabels.length][m_stickersDim];
		int[][] errors = new int[maxStages][2];
		
		for (int i =0; i < dataLabels.length; i += 1) {
			dataStickers[i] = intToSticker(dataLabels[i]);
		}
		for (int stageIndx = 0; stageIndx < maxStages-1; stageIndx += 1) {
			stage(data,dataLabels,dataStickers,eta);
			errors[stageIndx][0] = this.errorsDataSet(dataValidation,dataValidationLabels);
			errors[stageIndx][1] = this.errorsDataSet(data,dataLabels);
		}
		errors[maxStages-1][0] = this.errorsDataSet(dataValidation,dataValidationLabels);
		errors[maxStages-1][1] = this.errorsDataSet(data,dataLabels);
		return errors;
	}

	public float[][] learnWithErrorsCostsArray(float data[][], int dataLabels[],
											   float dataValidation[][], int dataValidationLabels[],
											   float eta, int maxStages) {
		int [][]dataStickers = new int [dataLabels.length][m_stickersDim];
		for(int stickerindex = 0; stickerindex < dataLabels.length; stickerindex ++){
			dataStickers[stickerindex] = intToSticker(dataLabels[stickerindex]);
		}
		float[][] errors = new float[maxStages][4];
		


		for (int stageIndx = 0; stageIndx < maxStages-1; stageIndx += 1) {
			errors[stageIndx][0] = this.errorsDataSet(dataValidation,dataValidationLabels);
			errors[stageIndx][1] = this.errorsDataSet(data,dataLabels);
			errors[stageIndx][2] = costFunction(dataValidation,dataValidationLabels);
			errors[stageIndx][3] = costFunction(data,dataLabels);
			stage(data,dataLabels,dataStickers,eta);
		}
		errors[maxStages-1][0] = this.errorsDataSet(dataValidation,dataValidationLabels);
		errors[maxStages-1][1] = this.errorsDataSet(data,dataLabels);
		errors[maxStages-1][2] = costFunction(dataValidation,dataValidationLabels);
		errors[maxStages-1][3] = costFunction(data,dataLabels);

		return errors;
	}

	
	private static String tabToString(float [] tab){
		String repTab = "[";
		for (int i =0; i < tab.length-1; i +=1) {
			repTab += "" + tab[i] +",";
		}
		repTab += "" + tab[ tab.length-1] + "]";
		return repTab;
	}
	private static String intSized(int n,int size){
		String toReturn = new String();
		int a= n;
		if (n==0){
			a = 1;
		}
		int num = 0;
		for (; a >0; a /= 10, num++);
		for (; num<size; num+=1){
			toReturn += " ";
		}
		toReturn += n;
		return toReturn;
	}
	private static String tabToString(int [] tab){
		String repTab = "[";
		for (int i =0; i < tab.length-1; i +=1) {
			int a = tab[i];

			repTab += intSized(tab[i],4) +",";
		}
		repTab += "" + intSized(tab[ tab.length-1],4) + "]";
		return repTab;
	}

	public String probaForPointString(float point []){
        return tabToString(probaForPoint(point));
    }

    public  String oneHotForLabel(int label){
	    return tabToString(intToSticker(label));
    }

	/**
	 * override of toString() function gives the perceptron's weights.
	 * @return
	 */
	@Override
	public String toString(){
		String perceptronRep = "......begin..Perceptron[" +m_stickersDim + "x" + m_pointDim +  "]......\n[";
		
		for (int perceptronIndex = 0; perceptronIndex < m_stickersDim-1; perceptronIndex+= 1) {
			for (int weightIndex = 0; weightIndex < m_pointDim-1; weightIndex += 1) {
				perceptronRep += "w{" + perceptronIndex + "," + weightIndex + "} = " +
									m_perceptronWeightsArray[perceptronIndex][weightIndex]+", ";
			}
			perceptronRep += "w{" + perceptronIndex + "," + (m_pointDim-1) + "} = " +
					m_perceptronWeightsArray[perceptronIndex][m_pointDim-1]+"\n";
		}
		for (int weightIndex = 0; weightIndex < m_pointDim-1; weightIndex += 1) {
			perceptronRep += "w{" + (m_stickersDim-1) + "," + weightIndex + "} = " +
								m_perceptronWeightsArray[m_stickersDim-1][weightIndex]+", ";
		}
		perceptronRep += "w{" + (m_stickersDim-1) + "," + (m_pointDim-1) + "} = " +
				m_perceptronWeightsArray[m_stickersDim-1][m_pointDim-1]+"]\n";

		perceptronRep += "......end....Perceptron[" +m_stickersDim + "x" + m_pointDim +  "]......";
		return perceptronRep;
	}

	/** FiveBienClassee
	 * fonction qui prend en paramètre les données(tableau de tableau de float) et leur label associé(tableau d'entiers)
	 * @param data
	 * @param datalabel
	 * @returnun tableau d'entier contenant l'indice des 5 images bien classé avec la plus faible probabilité
	 */
	public int[] FiveBienClassee(float data[][], int datalabel[]){
		int[] res = new int[5];
		ArrayList<Integer> BienClassee = new ArrayList<Integer>();
		for(int i = 0; i < data.length; i++){
			if(computeClass(data[i]) == datalabel[i]){
				BienClassee.add(i);
			}
		}
		int tabIndex = 0;
		while(tabIndex < 5) {
			float probamin = 1;
			Iterator<Integer> iter = BienClassee.iterator();
			while (iter.hasNext()) {
				int pointIndex = iter.next();
				float[] proba = probaForPoint(data[pointIndex]);
				if (proba[computeClass(data[pointIndex])] < probamin) {
					res[tabIndex] = pointIndex;
					probamin = proba[computeClass(data[pointIndex])];
				}
			}
			int pos = BienClassee.indexOf(res[tabIndex]);
			BienClassee.remove(pos);
			tabIndex++;
		 }
		return res;
	}

	/** LesPlusLoins
	 * fonction qui prend en parametre les données(tableau de tableau de float) les lablels associés(tableau de int)
	 * et la calsse que l'on souhaite observer(un int)
	 * @param data
	 * @param datalabel
	 * @param classe
	 * @return un tableau de int representant les indices des données mal classées de la classe donnée en parametre
	 * mais avec les plus forte probabilité de lui appartenir
	 */
	public int[] LesPlusLoins(float data[][], int datalabel[], int classe){
		int[] res = new int[5];
		ArrayList<Integer> MalClassee = new ArrayList<Integer>();
		for(int i = 0; i < data.length; i++){
			if(datalabel[i] == classe && computeClass(data[i]) != datalabel[i]){
				MalClassee.add(i);
			}
		}
		int tabIndex  = 0;
		int pointNumber = 5;
		if (pointNumber > MalClassee.size()){
			pointNumber = MalClassee.size();
		}
		while(tabIndex < pointNumber){
			float probaMax = 0;
			Iterator<Integer> iter = MalClassee.iterator();
			while(iter.hasNext()){
				int pointIndex = iter.next();
				float[] proba = probaForPoint(data[pointIndex]);
				if(proba[datalabel[pointIndex]] > probaMax){
					res[tabIndex] = pointIndex;
					probaMax = proba[datalabel[pointIndex]];
				}
			}
			int pos = MalClassee.indexOf(res[tabIndex]);
			MalClassee.remove((pos));
			tabIndex++;
		}
		return res;
	}

	public float [] stickerToData(float [] sticker){
		float[] point = new float[m_pointDim];
		float totalData = 0;
		for (int pixel = 0; pixel < m_pointDim; pixel += 1){
			for (int perceptronIndex = 0; perceptronIndex < m_stickersDim; perceptronIndex+= 1){
				point[pixel] += sticker[perceptronIndex] * m_perceptronWeightsArray[perceptronIndex][pixel];
				totalData += sticker[perceptronIndex];
			}
			point[pixel] /= totalData;
			totalData = 0;
		}
		return point;
	}

	
	
	public static void main(String[] args) {
		PerceptronMulti multPerceptron = new PerceptronMulti(2,3);
		System.out.println(multPerceptron);
		float point[] = new float[multPerceptron.m_pointDim];
		point[0] = 1;
		point[1] = 2.5f;
		System.out.println("point : "+ tabToString(point));
		System.out.println("proba classes : " + tabToString(multPerceptron.probaForPoint(point)));

		float proba[] = multPerceptron.probaForPoint(point);

		int stickerPoint[] = multPerceptron.intToSticker(2);
		multPerceptron.learnFromPoint(point,stickerPoint,1);
		
		
		proba = multPerceptron.probaForPoint(point);
		System.out.println("class : "+ tabToString(stickerPoint));
		System.out.println("proba classes : " + tabToString(proba));
		System.out.println(multPerceptron.computeClass(point));
		float [] sticker = new float [3];
		sticker[2] = 1;
		System.out.println(tabToString(multPerceptron.stickerToData(sticker)));

	}

}
