package perceptron;

import mnisttools.MnistReader;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Random;
import java.util.Vector;

public class ImageOnlinePerceptron {


    /* Les donnees */
    public static String path="src/resources/";
    public static String labelDB=path+"emnist-byclass-train-labels-idx1-ubyte";
    public static String imageDB=path+"emnist-byclass-train-images-idx3-ubyte";

    /* Parametres */
    // Na exemples pour l'ensemble d'apprentissage
    public static int Na = 500;
    // Nv exemples pour l'ensemble d'évaluation
    public static int Nv = 1000;
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

    public static int[][] convertPoint(float[] point, int sizeX, int sizeY,float seuil){
        int [][] image = new int[sizeY][sizeX];
        for (int y = 0; y < sizeY; y += 1){
            for( int x = 0; x < sizeX; x += 1){
                image[y][x] = point[1 + y * sizeX + x] >= seuil ? 1 : 0;
            }
        }
        return image;
    }

    public static float[][] convertPointf(float[] point, int sizeX, int sizeY){
        float [][] image = new float[sizeY][sizeX];
        for (int y = 0; y < sizeY; y += 1){
            for( int x = 0; x < sizeX; x += 1){
                image[y][x] = point[1 + y * sizeX + x];
            }
        }
        return image;
    }


    public static void displayImage(int[][] image){
        for (int y = image.length-1; y >= 0; y -= 1){
            for (int x = 0; x <image[y].length; x+=1){
                System.out.print(image[y][x] >= 1 ? '#' : '_');
            }
            System.out.print('\n');
        }
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

    private static float getBestCost(String fileName) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        float maxCost = -50000;

        try {
            String line;
            String[] numbers = null;
            float number;

            // read line by line till end of file
            while ((line = reader.readLine()) != null) {
                // split each line based on regular expression having
                // "any digit followed by one or more spaces".

                numbers = line.split(",");

                 number = Float.valueOf(numbers[0].trim());
                 if (maxCost < number){
                     maxCost = number;
                 }

                }
            } finally{
                reader.close();
        }
        return maxCost;
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
    private static String tabToString(int [] tab){
        String repTab = "[";
        for (int i =0; i < tab.length-1; i +=1) {
            repTab += "" + tab[i] +",";
        }
        repTab += "" + tab[ tab.length-1] + "]";
        return repTab;
    }


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


    public static PerceptronMulti genPerceptronPlusconfMat(int minLabel,int maxLabel, int Na, int Nv,
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

        perceptron.learnWithErrorsCostsArray(trainData, refs, valData, refsVal, eta, EPOCHMAX);
        System.out.println("# Perceptron done.");

        System.out.println(perceptron.stringConfusionMatrix(valData,refsVal));

        System.out.println("# Computation done" + filePrefix + ".");
        return perceptron;
    }


    public static void bestCosts(){
        for (int Na = 1000; Na <= 10000; Na += 100) {
            genPerceptronPlusCurves(10, 21, Na, 1000, "tenToTwentyOne" + Na,
                    150, 0.005f);

        }

        try {
            FileWriter fw = new FileWriter("bestCostRegardingNa.d");
            for (int Na = 1000; Na <= 10000; Na += 100) {
                fw.write("" + Na + " " + getBestCost("tenToTwentyOne"+Na+"ValidationCosts.d") + "\n");
            }
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static PerceptronMulti genPerceptronPlusMoyClasses(int minLabel,int maxLabel, int Na, int Nv,
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

        perceptron.learnWithErrorsCostsArray(trainData, refs, valData, refsVal, eta, EPOCHMAX);
        float probaMoy[]  = new float[Dim];
        float probaNow[]  = new float[Dim];
        for(int validationIndex = 0; validationIndex < Nv; validationIndex += 1) {
            probaNow = perceptron.probaForPoint(valData[validationIndex]);
            for (int j = 0; j < classe; j+= 1){
                probaMoy[j] += probaNow[j];
            }
        }
        System.out.print(probaMoy[0] / (float) Nv);
        for (int j = 1; j < classe; j+= 1){
            System.out.print(","+probaMoy[j] / (float) Nv);
        }
        System.out.println();

        System.out.println("# Computation done" + filePrefix + ".");
        return perceptron;
    }
    public static String tabToString(float [] tab){
        String repTab = "[";
        for (int i =0; i < tab.length-1; i +=1) {
            repTab += "" + tab[i] +",";
        }
        repTab += "" + tab[ tab.length-1] + "]";
        return repTab;
    }

    public static void  imageForClass(PerceptronMulti perceptron,int size, int classImage, String imageName){
        BufferedImage newImage = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        float[][] pixelf = convertPointf(perceptron.stickerToData(perceptron.intToSticker(classImage)),size,size);


        float max = pixelf[0][0];
        float min = max;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if(max < pixelf[i][j]){
                    max = pixelf[i][j];
                }
                else {
                    if (min > pixelf[i][j]){
                        min = pixelf[i][j];
                    }
                }
            }
        }
        int light;
        int pixel;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                light = Math.round(255*(( pixelf[i][j]-min)/(max-min)));
                pixel = light << 16 | light << 8 | light;
                newImage.setRGB(i,j,pixel);
            }
        }
        File outputFile = new File(imageName+classImage+".png");
        try {
            ImageIO.write(newImage, "png", outputFile);
        } catch (IOException e1) {

        }
    }

    public static void moyImageForAllClasses(float data[][],int refs [], int classSize, PerceptronMulti perceptron){
        Vector<BufferedImage> newImages =new Vector<BufferedImage>(0);

        float[][] pixelf = new float [classSize][data[0].length];
        int [] pixelsNs = new int[classSize];
        int pointClass = 0;
        for (int pointIndex =0; pointIndex < data.length; pointIndex+= 1){
            if (perceptron.computeClass(data[pointIndex]) != refs[pointIndex]){
                continue;
            }
            pointClass = refs[pointIndex];
            pixelsNs[pointClass]+=1;
            for (int pixelIndex = 0; pixelIndex < data[pointIndex].length; pixelIndex +=1){
                pixelf [pointClass][pixelIndex] += data[pointIndex][pixelIndex];
            }
        }
        int light;
        int pixel;
        for (pointClass = 0; pointClass < classSize; pointClass += 1){
            BufferedImage imageNowForClass = new BufferedImage (28, 28, BufferedImage.TYPE_INT_RGB);
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    light = Math.round(255*(( pixelf[pointClass][1+i*28 + j])/(pixelsNs[pointClass])));
                    pixel = light << 16 | light << 8 | light;
                    imageNowForClass.setRGB(i,j,pixel);
                }
            }
            try {
                ImageIO.write(imageNowForClass, "png",new File ("moyClass"+pointClass+".png"));
            } catch (IOException e1) {

            }
        }
    }

    public static void main(String[] args) throws IOException {
        Na = 6000;
        Nv = 1000;

        int minLabel = 10;
        int maxLabel = 35;
        String filePrefix = "hey";
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

        perceptron.learn(trainData, refs, .003f, 50);
        displayImage(BinariserImage(db.getImage(1),50));
        int size = db.getImage(1).length;
        moyImageForAllClasses(valData,refsVal,maxLabel - minLabel +1,perceptron);

        displayImage(convertPoint(ConvertImage(BinariserImage(db.getImage(1),5)),size, size,1));
        for (int i = 0; i < perceptron.m_stickersDim; i += 1){
            imageForClass(perceptron,28,i,"letterClass");
        }

    }
}