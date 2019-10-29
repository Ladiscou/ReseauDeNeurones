package perceptron;

import java.util.Random;

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
				m_perceptronWeightsArray[perceptronIndex][weightIndex] += eta * point[weightIndex] *(sticker[perceptronIndex]-probas[perceptronIndex]);
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
	
	
	public int stage(float data[][], int [] dataLabels, int [][] dataStickers, float eta) {
		int errorsNb = 0;
    	for (int i = 0; i < data.length; i +=1) {
    		if (computeClass(data[i]) != dataLabels[i]) {
    			learnFromPoint( data[i], dataStickers[i],eta);
    			errorsNb += 1;
    		}
    	}
		return errorsNb;
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
	
	public int[][] learnWithErrorsArray(float data[][], int dataLabels[], float dataTest[][], int dataTestLabels[], float eta, int maxStages) {
		int [][]dataStickers = new int [dataLabels.length][m_stickersDim];
		int[][] errors = new int[maxStages][2];
		
		for (int i =0; i < dataLabels.length; i += 1) {
			dataStickers[i] = intToSticker(dataLabels[i]);
		}
		for (int stageIndx = 0; stageIndx < maxStages; stageIndx += 1) {
			errors[stageIndx][0] = this.errorsDataSet(dataTest,dataTestLabels);
			errors[stageIndx][1] = stage(data,dataLabels,dataStickers,eta);
		}
		return errors;
	}

	public float[][] learnWithErrorsCostsArray(float data[][], int dataLabels[], float dataTrain[][], int dataTrainLabels[],
			float eta, int maxStages) {
		int [][]dataStickers = new int [dataLabels.length][m_stickersDim];
		float[][] errors = new float[maxStages][4];
		

		
		for (int stageIndx = 0; stageIndx < maxStages; stageIndx += 1) {
			errors[stageIndx][0] = this.errorsDataSet(dataTrain,dataTrainLabels);
			errors[stageIndx][1] = stage(data,dataLabels,dataStickers,eta);
			errors[stageIndx][2] = costFunction(dataTrain,dataLabels);
			errors[stageIndx][3] = costFunction(data,dataTrainLabels);
		}
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
	
	private float costFunction(float data[][], int [] dataLabels) {
		float Etotal = 0;
		float[] probas;
		for (int imageIndex = 0; imageIndex < data.length; imageIndex += 1) {
			probas[]
			Etotal += (float) Math.log(dotProd(data[imageIndex],m_perceptronWeightsArray[dataLabels[imageIndex]])));
		}
		return Etotal/data.length;
	}

	
	
	public static void main(String[] args) {
		PerceptronMulti multPerceptron = new PerceptronMulti(2,3);
		System.out.println(multPerceptron);
		float point[] = new float[multPerceptron.m_pointDim];
		point[0] = 1;
//		point[2] = 2.5f;
		System.out.println("point : "+ tabToString(point));
		System.out.println("proba classes : " + tabToString(multPerceptron.probaForPoint(point)));

		float proba[] = multPerceptron.probaForPoint(point);
		
		int stickerPoint[] = multPerceptron.intToSticker(0);
		multPerceptron.learnFromPoint(point,stickerPoint,1);
		
		
		proba = multPerceptron.probaForPoint(point);
		System.out.println("class : "+ tabToString(stickerPoint));
		System.out.println("proba classes : " + tabToString(proba));
		System.out.println(multPerceptron.computeClass(point));
		
		
		// TODO Auto-generated method stub

	}

}
