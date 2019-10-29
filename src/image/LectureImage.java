package image;
import java.util.ArrayList;
import java.util.List;
import mnisttools.MnistReader;

public class LectureImage {
	/**
	 * @param args
	 */
	
	public static int[][] binarisation(int[][] image, int seuil)
	{
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
	
	private static int compteAllumeeLigne(int [][] image, int ligne) {
		int compteur = 0;
		for (int x = 0; x < image[ligne].length; x += 1) {
			compteur += image[ligne][x];
		}
		return compteur;
	}
	
	private static int[] comptePixelAllumeePourChaqueLigne(int [][] image) {
		int [] tab = new int[image.length];
		for(int y = 0; y < image.length; y += 1) {
			tab[y] = compteAllumeeLigne(image,y);
		}
		return tab;
	}
	
	private static int maxList(int [] tab) {
		int max = tab[0];
		for (int i = 0; i < tab.length; i +=1) {
			max = tab[i]>max? tab[i] : max;
		}
		return max;
	}
	
	
	private static int maxLineImage(int[][] image, int seuil) {
		return maxList(comptePixelAllumeePourChaqueLigne(binarisation(image,seuil)));
	}
	
	public static void main(String[] args) {
		String path="src/resources/"; // TODO : indiquer le chemin correct
		String labelDB=path+"train-labels-idx1-ubyte";
	    String imageDB=path+"train-images-idx3-ubyte";
	    // Creation de la base de donnees
	    MnistReader db = new MnistReader(labelDB, imageDB);
	    // Acces a la premiere image
	    int idx = 1; // une variable pour stocker l'index
	                // Attention la premiere valeur est 1.
	    int [][] image = db.getImage(idx); /// On recupere la premiere l'image numero idx
	    // Son etiquette ou label
	    int label = db.getLabel(idx);
	    // Affichage du label
	    System.out.print("Le label est "+ label+"\n"); 
	    // note: le caractère \n est le 'retour charriot' (retour a la ligne).
	    // Affichage du nombre total d'image
	    System.out.print("Le total est "+ db.getTotalImages()+"\n");
	    /* A vous de jouer pour la suite */
	    
	    int [] maxColumns = new int[image.length];
	    int [] minColumns = new int[image.length];
	    
	    int [] maxLines = new int[image[0].length];
	    int [] minLines = new int[image[0].length];
	    
	    
	    int maxLine;
	    
	    int minLine;
	    
	    
	    int length = db.getTotalImages();
	    List<Integer> eights = new ArrayList<Integer>();
	    List<Integer> ones = new ArrayList<Integer>();

	    for (int index = 1; index < length; index += 1) {
	    	if (db.getLabel(index) == 8) {
	    		eights.add( maxLineImage(db.getImage(index),50));
	    	}else {
	    		if(db.getLabel(index) == 1) {
	    			ones.add( maxLineImage(db.getImage(index),50));
	    		}
	    	}
	    }
	    
	    float moyenne1s = 0;
	    for (int value : ones) {
	    	moyenne1s += value;
	    }
	    moyenne1s /= ones.size();
	    System.out.println(moyenne1s);
	    

	    float moyenne8s = 0;
	    for (int value : eights) {
	    	moyenne8s += value;
	    }
	    moyenne8s /= ones.size();
	    System.out.println(moyenne8s);
	    
	    for (int y = 0; y < image.length; y += 1) 
	    {
	    	maxLine = image[y][0];
	    	minLine = image[y][0];
	    	for(int x = 0; x < image[y].length; x += 1) 
	    	{
	    		if (maxLine < image[y][x]) 
	    		{
	    			maxLine = image[y][x];
	    		}
	    		else 
	    		{
	    			if(minLine > image[y][x]) 
	    			{
	    				minLine = image[y][x];
	    			}
	    		}
	    		
	    		if (maxColumns[x] < image[y][x])
	    		{
	    			maxColumns[x] = image[y][x];
	    		}
	    		else
	    		{
	    			if(minColumns[x] > image[y][x]) 
	    			{
	    				minColumns[x] = image[y][x];
	    			}
	    		}
	    	}
	    	maxLines[y] = maxLine;
	    	minLines[y] = minLine;
	    }
	    
	    
	    
	}

	

}
