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
    public static final int Na = 6000;
    // Nv exemples pour l'ensemble d'évaluation
    public static final int Nv = 1000;
    //Nt exemple pour l'ensemble de test
    public static final int Nt = 1000;
    // Nombre d'epoque max
    public final static int EPOCHMAX=50;
    // Classe positive (le reste sera considere comme des ex. negatifs):
    public static int  classe = 12;
    // index actuel dans la base de donnée
    private static int imageIndex = 0;


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

    /**
     * fonction qui crée un fichier et y ecrit les donnée
     * @param fileNames
     * @param data
     */
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

    /**
     * fonction cree pour la question avec Nt ecris dans un fichier  deux colone representant un tableau de x et un de y
     * @param fileNames
     * @param data
     * @param eta
     */
    private static void dataFilesWritertest(String fileNames,  float data[], float eta[]) {
        try {
            FileWriter fw = new FileWriter(fileNames + ".d");
            for (int i = 0; i < data.length; i++) {
                fw.write("" + eta[i] + "  " + data[i] + 0 + "\n");
            }
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void gnuplotFileWriter(String plotName,String[] fileNames,float eta){
        try {
            FileWriter fw = new FileWriter(plotName + ".gnu");
            fw.write("set terminal svg size 2000,1000 \nset output 'histogram");
            fw.write(""+plotName+eta+""+classe);
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

    private static void gnuplotFileWritertest(String plotName, String fileName){
        try {
            FileWriter fw = new FileWriter(plotName + ".gnu");
            fw.write("set terminal svg size 2000,1000 \nset output 'histogram");
            fw.write("" + plotName + "" + classe);
            fw.write("Multi.svg'\nset title \"Na = " + Na + " Nv = " + Nv + "\" \n");
            fw.write("set grid\nset style data linespoints\nplot");
            fw.write("'"+fileName + ".d',");
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

    /**
     * fonction qui initialise un tableau de data dans la base de donnée actuel
     * @param minLabel
     * @param maxLabel
     * @param data
     * @param dataDim
     * @param refs
     * @param db
     */
    public static void dataGenerator(int minLabel, int maxLabel, float [][] data, int dataDim,
                                    int [] refs, MnistReader db){
        int length = db.getTotalImages();
        for (int trainDataIndex = 0; trainDataIndex < refs.length && imageIndex < length; imageIndex += 1) {
            int imageLabel = db.getLabel(imageIndex+1);
            if (imageLabel >=minLabel && imageLabel <= maxLabel){
                data[trainDataIndex] = ConvertImage(BinariserImage(db.getImage(imageIndex+1),50));
                refs [trainDataIndex] = db.getLabel(imageIndex+1)-10;
                trainDataIndex += 1;
            }
        }
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
        imageIndex = 0;
        classe = maxLabel - minLabel + 1;
        System.out.println("# Load the database for " + filePrefix + " !");
        MnistReader db = new MnistReader(labelDB, imageDB);
        int Dim = (db.getImage(1).length * db.getImage(1)[0].length)+1;

        float[][] trainData = new float[Na][Dim];

        int [] refs= new int[Na];


        dataGenerator(minLabel,maxLabel,trainData,Dim,refs,db);

        System.out.println("# Built train for "+filePrefix + ".");

        System.out.println("# Load validation for digits ");

        float[][] valData = new float [Nv][Dim];
        int [] refsVal = new int[Nv];

        dataGenerator(minLabel,maxLabel,valData,Dim,refsVal,db);

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

    public static PerceptronMulti genPerceptronPlusCurvesPlusClassement(int minLabel,int maxLabel, int Na, int Nv,
                                                          String filePrefix,int stagesNumber, float eta) {
        imageIndex = 0;
        classe = maxLabel - minLabel + 1;
        System.out.println("# Load the database for " + filePrefix + " !");
        MnistReader db = new MnistReader(labelDB, imageDB);
        int Dim = (db.getImage(1).length * db.getImage(1)[0].length) + 1;

        float[][] trainData = new float[Na][Dim];

        int[] refs = new int[Na];


        dataGenerator(minLabel, maxLabel, trainData, Dim, refs, db);

        System.out.println("# Built train for " + filePrefix + ".");

        System.out.println("# Load validation for digits ");

        float[][] valData = new float[Nv][Dim];
        int[] refsVal = new int[Nv];

        dataGenerator(minLabel, maxLabel, valData, Dim, refsVal, db);

        System.out.println("# Built validation for " + filePrefix + ".");

        PerceptronMulti perceptron = new PerceptronMulti(Dim, classe);

        float[][] errorsCurvePlots = perceptron.learnWithErrorsCostsArray(trainData, refs, valData, refsVal, eta, EPOCHMAX);
        System.out.println("# Perceptron done.");
        System.out.println("les 5 biens classées sont: ");
        int [] five = perceptron.FiveBienClassee(valData, refsVal);
        for(int i = 0; i < 5; i++) {
            int point = perceptron.FiveBienClassee(valData, refsVal)[i];
            System.out.print(point + " sa proba  pour" + refsVal[point] + "est: ");
            float[] proba = perceptron.probaForPoint(valData[point]);
            System.out.println(proba[refsVal[point]]);
        }
        System.out.println("les mal classés sont:");
        for(int i = 0; i < 5; i ++){
            int point = perceptron.LesPlusLoins(valData, refsVal, i)[0];
            System.out.print(point + " sa proba est: ");
            float [] proba = perceptron.probaForPoint(valData[point]);
            System.out.print(proba[refsVal[point]]);
            System.out.println(" estimer pour " + perceptron.computeClass(valData[point]));

        }


        System.out.println("Validation accuracy : " + 100.f * (1.f - (float) (errorsCurvePlots[EPOCHMAX - 1][0]) / Nv) +
                "%, Training accuracy : " + 100.f * (1.f - (float) errorsCurvePlots[EPOCHMAX - 1][1] / Na) + '%');

        String[] fileNames = new String[4];
        fileNames[0] = filePrefix + "ValidationErrors";
        fileNames[1] = filePrefix + "TrainingErrors";

        fileNames[2] = filePrefix + "ValidationCosts";
        fileNames[3] = filePrefix + "TrainingCosts";

        dataFilesWriter(fileNames, errorsCurvePlots);

        String[] errorsNames = new String[2];
        errorsNames[0] = fileNames[0];
        errorsNames[1] = fileNames[1];

        String[] costsNames = new String[2];
        costsNames[0] = fileNames[2];
        costsNames[1] = fileNames[3];


        gnuplotFileWriter(filePrefix + "Errors", errorsNames, eta);

        gnuplotFileWriter(filePrefix + "Costs", costsNames, eta);
        System.out.println("# Computation done" + filePrefix + ".");
        return perceptron;
    }

    public static float minTableau(float data[]){
        float min = data[0];
        for(int i = 1; i < data.length; i++){
            if (data[i] < min){
                min = data[i];
            }
        }
        return min;
    }



    public static void main(String[] args) {
        /*  //code pour generer it perceptron avec eta allant de 0.001 a 0.001 * it
        int it = 100;
        float eta = 0.001f;
        float [] etas = new float[it];
        float [] Nverror = new float[it];
        int maxLabel = 21;
        int minLabel = 10;
        MnistReader db = new MnistReader(labelDB, imageDB);
        int Dim = (db.getImage(1).length * db.getImage(1)[0].length)+1;

        for(int i = 0; i < it; i++) {
            imageIndex = 0;
            classe = maxLabel - minLabel + 1;
            float[][] trainData = new float[Na][Dim];
            int[] refs = new int[Na];
            dataGenerator(minLabel, maxLabel, trainData, Dim, refs, db);
            float[][] valData = new float[Nv][Dim];
            int[] refsVal = new int[Nv];
            dataGenerator(minLabel, maxLabel, valData, Dim, refsVal, db);
            PerceptronMulti perceptron = new PerceptronMulti(Dim,classe);
            float[][] errors = perceptron.learnWithErrorsCostsArray(trainData, refs, valData, refsVal, eta, EPOCHMAX);
            float [] lesErreur = new float[errors.length]; // ici on recupere le nombre d'erreur sur Nv
            for(int j = 0; j < errors.length; j ++){
                lesErreur[j] = errors[j][0];
            }
            etas[i] = eta;
            Nverror[i] = minTableau(lesErreur);
            eta += 0.001f;
        }

        System.out.println(etas[5]);
        System.out.println(Nverror[2]);
        dataFilesWritertest("test", Nverror, etas); // fonction qui genere le .d
        gnuplotFileWritertest("Nv_en_fonction_de_eta", "test"); // fonction qui genere le .gnu

         //ici on initialise Nt et teste avec le perceptron adapté le nombre d'erreur obtenue
        PerceptronMulti tavu = genPerceptronPlusCurves(10, 21, 10000, 1000, "tenToTwentyOne", EPOCHMAX, 0.003f);
        MnistReader db = new MnistReader(labelDB, imageDB);
        int Dim = (db.getImage(1).length * db.getImage(1)[0].length)+1;
        float [][] testData = new float[Nt][Dim];
        int [] refs= new int[Nt];
        dataGenerator(10 , 21, testData, Dim, refs, db);
        int erreurT = tavu.errorsDataSet(testData,refs);
        System.out.println(erreurT);

         */
        PerceptronMulti perceptron = genPerceptronPlusCurvesPlusClassement(10, 21, 6000, 1000, "tenToTwentyOne", EPOCHMAX, 0.003f);

    }
}