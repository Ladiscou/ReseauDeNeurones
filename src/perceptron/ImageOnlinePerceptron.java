package perceptron;

import mnisttools.MnistReader;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class ImageOnlinePerceptron {

    /* Les donnees */
    public static String path="src/resources/";
    public static String labelDB=path+"train-labels-idx1-ubyte";
    public static String imageDB=path+"train-images-idx3-ubyte";

    /* Parametres */
    // Na exemples pour l'ensemble d'apprentissage
    public static final int Na = 1000;
    // Nv exemples pour l'ensemble d'évaluation
    public static final int Nv = 500;
    // Nombre d'epoque max
    public final static int EPOCHMAX=50;
    // Classe positive (le reste sera considere comme des ex. negatifs):
    public static int  classe = 5;

    // Générateur de nombres aléatoires
    public static int seed = 1234;
    public static Random GenRdm = new Random(seed);
    

    /*
    *  BinariserImage : 
    *      image: une image int à deux dimensions (extraite de MNIST)
    *      seuil: parametre pour la binarisation
    *
    *  on binarise l'image à l'aide du seuil indiqué
    *
    */
    public static int[][] BinariserImage(int[][] image, int seuil) {
    	for (int y = 0; y < image.length; y += 1) 
	    {
	    	for(int x = 0; x < image[y].length; x += 1) 
	    	{
	    		image [y][x] = image[y][x] > seuil? 
	    				1:
	    				0;
	    	}
	    }
		return image;
    }

    /*
    *  ConvertImage : 
    *      image: une image int binarisée à deux dimensions
    *
    *  1. on convertit l'image en deux dimension dx X dy, en un tableau unidimensionnel de tail dx.dy
    *  2. on rajoute un élément en première position du tableau qui sera à 1
    *  La taille finale renvoyée sera dx.dy + 1
    *
    */
    public static float[] ConvertImage(int[][] image) {
            float[] uniDim = new float[image.length * image[0].length+1];
            int width = image.length;
            int height = image[0].length;
            uniDim[0] = 1;
            for (int y = 0; y < width; y+=1) {
            	for (int x = 0; x < height; x+=1) {
            		uniDim[y * width + x + 1] = image[y][x];
            	}
            }
            return uniDim;
    }

    /*
    *  InitialiseW :
    *      sizeW : la taille du vecteur de poids
    *      alpha : facteur à rajouter devant le nombre aléatoire
    *
    *  le vecteur de poids est crée et initialisé à l'aide d'un générateur
    *  de nombres aléatoires.
    */
    public static float[] InitialiseW(int sizeW, float alpha) {
            // TODO
    	float[] w = new float[sizeW];
    	for (int i = 0; i < sizeW; i += 1) {
    		w[i] = alpha * GenRdm.nextFloat();
    	}
    	return w;
    }

    private static void dataFilesWriter(String[] fileNames, float data[][]) {
        for (int j = 0; j < fileNames.length; j += 1) {
            try {
                FileWriter fw = new FileWriter(fileNames[j]+".d");
                for (int i = 0; i < data.length; i += 1) {
                    fw.write("" + data[i][j] + "," + 0 + "\n");
                }
                fw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


    private static void gnuplotFileWriter(String plotName,String[] fileNames,float eta){
        try {
            FileWriter fw = new FileWriter(plotName + ".gnu");
            fw.write("set terminal svg size 2000,1000 \nset output 'histogram");

            fw.write(""+eta);
            fw.write("Multi.svg'\nset title \"Na = "+Na+" Nv = "+Nv+"\" \n");
            fw.write("set grid\nset style data linespoints\nplot");
            for (int i = 0; i < fileNames.length-1; i+= 1){
                fw.write("'"+fileNames[i] + ".d',");
            }
            fw.write("'"+fileNames[fileNames.length-1]+".d'\n");
            fw.close();
        }
        catch(IOException e){
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
    	System.out.println("# Load the database !");
        MnistReader db = new MnistReader(labelDB, imageDB);
        int Dim = db.getImage(1).length * db.getImage(1)[0].length+1;
        
        float[][] trainData = new float[Na][Dim];
        
        int [] refs = new int[Na];
        for (int imageIndex = 0; imageIndex < Na; imageIndex += 1) {
        	trainData[imageIndex] = ConvertImage(BinariserImage(db.getImage(imageIndex+1),50));
        	refs [imageIndex] = db.getLabel(imageIndex+1);
        }
        System.out.println("# Built train for digits ");
        
        System.out.println("# Load validation for digits ");
        float[][] valData = new float[Nv][Dim];
        int [] refsVal = new int[Nv];
        for (int imageIndex = Na; imageIndex < Na+Nv; imageIndex += 1) {
        	valData[imageIndex-Na] = ConvertImage(BinariserImage(db.getImage(imageIndex+1),125));
        	refsVal [imageIndex-Na] = db.getLabel(imageIndex+1);
        }
        System.out.println("# Built validation for digits ");
        
        PerceptronMulti numberTeller = new PerceptronMulti(Dim,10);
        System.out.println(numberTeller.probaForPointString(trainData[2]));
        System.out.println(numberTeller.oneHotForLabel(refs[2]));

        System.out.println(numberTeller.probaForPointString(trainData[Na/2]));
        System.out.println(numberTeller.oneHotForLabel(refs[Na/2]));

        float eta = 0.0005f;
        float [][] errorsCurvePlots = numberTeller.learnWithErrorsCostsArray(trainData, refs, valData, refsVal, eta, EPOCHMAX);
        System.out.println("# Perceptron done.");

        System.out.println("Validation accuracy : "+ 100.f * (1.f- (float)(errorsCurvePlots[EPOCHMAX-1][0]) /Nv) +
        					"%, Training accuracy : " + 100.f *(1.f- (float)errorsCurvePlots[EPOCHMAX-1][1]/Na) + '%');

        String [] fileNames = new String[4];
        fileNames[0] = "validationErrors";
        fileNames[1] = "trainingErrors";

        fileNames[2] = "validationCosts";
        fileNames[3] = "trainingCosts";

        dataFilesWriter(fileNames,errorsCurvePlots);

        String [] errorsNames = new String[2];
        errorsNames[0] = fileNames[0];
        errorsNames[1] = fileNames[1];

        String [] costsNames = new String[2];
        costsNames[0] = fileNames[2];
        costsNames[1] = fileNames[3];


        gnuplotFileWriter("errors",errorsNames,eta);

        gnuplotFileWriter("costs",costsNames,eta);

        System.out.println(numberTeller.probaForPointString(trainData[2]));
        System.out.println(numberTeller.oneHotForLabel(refs[2]));

        System.out.println(numberTeller.probaForPointString(trainData[Na/2]));
        System.out.println(numberTeller.oneHotForLabel(refs[Na/2]));
        System.out.println(numberTeller);
        System.out.println("# Computation done.");

    }
}