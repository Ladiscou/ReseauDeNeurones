package perceptron;

import java.util.Iterator;
import java.util.Random;
import java.util.ArrayList;

public class PerceptronMulti {
	public int m_pointDim = 12;
	public int m_stickersDim = 10;
	private float m_perceptronWeightsArray [][] = new float[m_stickersDim][m_pointDim];
	
	
	
	public PerceptronMulti(int pointDim,int stickersDim, float weights[][]) {
		m_pointDim = pointDim;
		m_stickersDim = stickersDim;
		m_perceptronWeightsArray = weights;
	}
	
	private void genRandomWeights(float alpha) {
		Random randGen= new Random();
		
		for (int i = 0; i < m_stickersDim; i += 1) {
			for (int j = 0; j < m_pointDim; j += 1) {
				m_perceptronWeightsArray[i][j] = randGen.nextFloat() * alpha;
			}
		}
	}
	
	public PerceptronMulti(int pointDim,int stickersDim) {
		m_pointDim = pointDim;
		m_stickersDim = stickersDim;
		m_perceptronWeightsArray = new float[stickersDim][pointDim];
		genRandomWeights(1.f/(pointDim));
	}

	public int[] intToSticker(int a) {
		int [] sticker = new int[m_stickersDim];
		for(int i = 0; i < m_stickersDim; i ++){  //rajout de la boucle pour initialiser les autres valeurs a 0
			sticker[i] = 0;
		}
		sticker[a%m_stickersDim] = 1;
		return sticker;
	}
	
	private float dotProd(float[] a, float[] b) {
		float sum = 0;
		for (int i =0; i < m_pointDim; i += 1) {
			sum += a[i] * b[i];
		}
		return sum;
	}
	
	
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
	
	
	
	
	private void learnFromPoint(float point[], int [] sticker,float eta) {
		float[] probas = probaForPoint(point);
		for (int perceptronIndex =0; perceptronIndex < m_stickersDim; perceptronIndex+= 1) {
			for (int weightIndex = 0; weightIndex < m_pointDim; weightIndex+=1) {
				m_perceptronWeightsArray[perceptronIndex][weightIndex] -=
						eta * point[weightIndex] *(probas[perceptronIndex] - sticker[perceptronIndex]);
			}
		}
	}
	
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
	
	
	public void stage(float data[][], int [] dataLabels, int [][] dataStickers, float eta) {
		int errorsNb = 0;
    	for (int i = 0; i < data.length; i +=1) {
			learnFromPoint(data[i], dataStickers[i], eta);
		}
	}
	
	public void learn(float data[][], int dataLabels[], float eta, int maxStages) {
		int [][]dataStickers = new int [dataLabels.length][m_stickersDim];
		for (int i =0; i < dataLabels.length; i += 1) {
			dataStickers[i] = intToSticker(dataLabels[i]);
		}
		for (int stageIndx = 0; stageIndx < maxStages; stageIndx += 1) {
			stage(data,dataLabels,dataStickers,eta);
		}
	}
	
	public int errorsDataSet(float data[][], int dataLabels[]) {
		int nbErrors = 0;
		for (int i = 0; i < data.length; i += 1) {
			if (this.computeClass(data[i]) != dataLabels[i]) {
				nbErrors += 1;
			}
		}
		return nbErrors;
	}

	private float costFunction(float data[][], int [] dataLabels) {
		float Etotal = 0;
		for (int imageIndex = 0; imageIndex < data.length; imageIndex += 1) {
			Etotal += (float) Math.log(probaForPoint(data[imageIndex])[dataLabels[imageIndex]]);
		}
		return Etotal/data.length;
	}
	/*
	 * @return a matrix, with numbers in slots:
	 * 					which lines (first index) indexes, are the class number the perceptron gave.
	 * 					which columns (second index) indexes, are the class number they really are.
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

	public String stringConfusionMatrix(float data [][], int [] dataLabels){
		int[][] confMat = confusionMatrix(data,dataLabels);
		String ret = tabToString(confMat[0]);
		for (int i = 1; i < m_stickersDim; i+=1){
			ret += '\n'+tabToString(confMat[i]);
		}
		return ret;
	}

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

	private static String tabToString(int [] tab){
		String repTab = "[";
		for (int i =0; i < tab.length-1; i +=1) {
			repTab += "" + tab[i] +",";
		}
		repTab += "" + tab[ tab.length-1] + "]";
		return repTab;
	}

	public String probaForPointString(float point []){
        return tabToString(probaForPoint(point));
    }

    public  String oneHotForLabel(int label){
	    return tabToString(intToSticker(label));
    }

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
		int u = 0;
		while(u < 5) {
			float probamin = 1;
			Iterator<Integer> iter = BienClassee.iterator();
			while (iter.hasNext()) {
				int a = iter.next();
				float[] proba = probaForPoint(data[a]);
				if (proba[computeClass(data[a])] < probamin) {
					res[u] = a;
					probamin = proba[computeClass(data[a])];
				}
			}
			int pos = BienClassee.indexOf(res[u]);
			BienClassee.remove(pos);
			u++;
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
			if(computeClass(data[i]) != datalabel[i] && datalabel[i] == classe){
				MalClassee.add(i);
			}
		}
		int u  = 0;
		while(u < 5){
			float probaMax = 0;
			Iterator<Integer> iter = MalClassee.iterator();
			while(iter.hasNext()){
				int a = iter.next();
				float[] proba = probaForPoint(data[a]);
				if(proba[datalabel[a]] > probaMax){
					res[u] = a;
					probaMax = proba[datalabel[a]];
				}
			}
			int pos = MalClassee.indexOf(res[u]);
			MalClassee.remove((pos));
			u++;
		}
		return res;
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


	}

}
