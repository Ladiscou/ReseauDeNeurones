package perceptron;

import mnisttools.MnistReader;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class ImageOnlinePerceptron {


    /* Les donnees */
    public static String path="src/resources/";
    public static String labelDB=path+"emnist-byclass-train-labels-idx1-ubyte";
    public static String imageDB=path+"emnist-byclass-train-images-idx3-ubyte";

    /* Parametres */
    // Na exemples pour l'ensemble d'apprentissage
    public static final int Na = 500;
    // Nv exemples pour l'ensemble d'évaluation
    public static final int Nv = 500;
    // Nombre d'epoque max
    public final static int EPOCHMAX=50;
    // Classe positive (le reste sera considere comme des ex. negatifs):
    public static int  classe = 12;

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
            fw.write(""+plotName+eta+""+classe);
            fw.write("Multi.svg'\nset title \"Na = "+Na+" Nv = "+Nv+"\" \n");
            fw.write("set object 1 rectangle from screen 0,0 to screen 1,1 fillcolor rgb\"white\" behind");
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
    private static String tabToString(int [] tab){
        String repTab = "[";
        for (int i =0; i < tab.length-1; i +=1) {
            repTab += "" + tab[i] +",";
        }
        repTab += "" + tab[ tab.length-1] + "]";
        return repTab;
    }


    public static int dataGenerator(int minLabel, int maxLabel, int startIndex, float [][] data, int dataDim,
                                    int [] refs, MnistReader db){
        int imageIndex = startIndex;
        int length = db.getTotalImages();
        for (int trainDataIndex = 0; trainDataIndex < refs.length && imageIndex < length; imageIndex += 1) {
            int imageLabel = db.getLabel(imageIndex+1);
            if (imageLabel >=minLabel && imageLabel <= maxLabel){
                data[trainDataIndex] = ConvertImage(BinariserImage(db.getImage(imageIndex+1),50));
                refs [trainDataIndex] = db.getLabel(imageIndex+1)-10;
                trainDataIndex += 1;
            }
        }

        return imageIndex;
    }

    /**
     *
     * @param minLabel label extracted begin
     * @param maxLabel label extracted end
     *
     * @param Na Size of the training data
     * @param Nv Size of the validation data
     *
     * @param filePrefix general name of the files generated (data for curves & gnuplotter)
     *
     * @param stagesNumber number of stages.
     * @param eta training rate
     *
     * @return PerceptronMulti trained with data perceptron.
     * @creates files with the evolution data of the perceptron.
     */
    public static PerceptronMulti genPerceptronPlusCurves(int minLabel,int maxLabel, int Na, int Nv,
                                                           String filePrefix,int stagesNumber, float eta){
        classe = maxLabel - minLabel + 1;
        System.out.println("# Load the database for " + filePrefix + " !");
        MnistReader db = new MnistReader(labelDB, imageDB);
        int Dim = (db.getImage(1).length * db.getImage(1)[0].length)+1;

        float[][] trainData = new float[Na][Dim];

        int [] refs= new int[Na];

        int imageIndex = 0;

        imageIndex = dataGenerator(minLabel,maxLabel,imageIndex,trainData,Dim,refs,db);

        System.out.println("# Built train for "+filePrefix + ".");

        System.out.println("# Load validation for digits ");

        float[][] valData = new float [Nv][Dim];
        int [] refsVal = new int[Nv];

        imageIndex = dataGenerator(minLabel,maxLabel,imageIndex,valData,Dim,refsVal,db);

        System.out.println("# Built validation for "+filePrefix+".");

        PerceptronMulti perceptron = new PerceptronMulti(Dim,classe);

        float [][] errorsCurvePlots = perceptron.learnWithErrorsCostsArray(trainData, refs, valData, refsVal, eta, EPOCHMAX);
        System.out.println("# Perceptron done.");

        System.out.println("Validation accuracy : "+ 100.f * (1.f- (float)(errorsCurvePlots[EPOCHMAX-1][0]) /Nv) +
                "%, Training accuracy : " + 100.f *(1.f- (float)errorsCurvePlots[EPOCHMAX-1][1]/Na) + '%');

        String [] fileNames = new String[4];
        fileNames[0] = filePrefix + "ValidationErrors";
        fileNames[1] = filePrefix + "TrainingErrors";

        fileNames[2] = filePrefix + "ValidationCosts";
        fileNames[3] = filePrefix + "TrainingCosts";

        dataFilesWriter(fileNames,errorsCurvePlots);

        String [] errorsNames = new String[2];
        errorsNames[0] = fileNames[0];
        errorsNames[1] = fileNames[1];

        String [] costsNames = new String[2];
        costsNames[0] = fileNames[2];
        costsNames[1] = fileNames[3];


        gnuplotFileWriter(filePrefix + "Errors",errorsNames,eta);

        gnuplotFileWriter(filePrefix + "Costs",costsNames,eta);
        System.out.println("# Computation done" + filePrefix + ".");
        return perceptron;
    }




    public static void main(String[] args) {
        genPerceptronPlusCurves(10,21,500,100 ,"tenToTwentyOne",
                50,0.001f);
        genPerceptronPlusCurves(10,21,500,100 ,"tenToTwentyOne",
                50,0.005f);
        genPerceptronPlusCurves(10,21,500,100 ,"tenToTwentyOne",
                50,0.01f);
        genPerceptronPlusCurves(10,21,500,100 ,"tenToTwentyOne",
                50,0.05f);
        genPerceptronPlusCurves(10,21,500,100 ,"tenToTwentyOne",
                50,0.05f);

    }
}