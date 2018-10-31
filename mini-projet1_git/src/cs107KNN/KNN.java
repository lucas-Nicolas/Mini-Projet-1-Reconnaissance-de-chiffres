package cs107KNN;

import cs107KNN.Helpers;



public class KNN {
	public static void main(String[] args) {
		int TESTS = 700;
		int K =1;
		byte[][][] trainImages =
				parseIDXimages(Helpers.readBinaryFile("datasets/reduced50Kto5K_images_fictive"));
		byte[] trainLabels =
				parseIDXlabels(Helpers.readBinaryFile("datasets/reduced50Kto5K_labels_fictive"));

		byte[][][] testImages = parseIDXimages(Helpers.readBinaryFile("datasets/10k_images_test"));

		byte[] testLabels = parseIDXlabels(Helpers.readBinaryFile("datasets/10k_labels_test"));

		byte[] predictions = new byte[TESTS];

		long start = System.currentTimeMillis();

		for (int i = 0; i < TESTS; i++) {
			predictions[i] = knnClassify(testImages[i], trainImages, trainLabels, K);
		}
		long end = System.currentTimeMillis();
		Helpers.show("Test", testImages, predictions, testLabels, 20, 35) ;
		System.out.println("Taux de précision : "+accuracy(predictions, testLabels)+"%");



		double time = (end - start) / 1000d;

		System.out.println("temps de calcul : " + time);
		System.out.println("Temps par image : "+ time/TESTS);
	}
		
	/**
	 * Méthode pour rendre signés tous les bytes non signés d'une image ou d'une étiquette IDX
	 * @param //les sets d'images que l'on va modifier
	 * @return le set d'images avec pixels correctement utilisés, directement exploitable par le parseur d'images
	 */
	public static byte[] signConverter(byte[] toConvert) {
			//On met le dataset dans un tableau
		//On crée le tableau des valeurs que l'on va signer
		for(int i = 16; i < toConvert.length; ++i) {
			toConvert[i] = (byte) ((toConvert[i] & 0xFF) - 128);	//On applique la conversion exactement comme proposée dans l'énnoncé 3.4.2
		}
		
		return toConvert;	//On retourne le tableau converti correctement que l'on pourra utiliser pour avoir les images
	}
	
	/**
	 * Composes four bytes into an integer using big endian convention.
	 *
	 * @param //bXToBY The byte containing the bits to store between positions X and Y
	 * 
	 * @return the integer having form [ b31ToB24 | b23ToB16 | b15ToB8 | b7ToB0 ]
	 */
	public static int extractInt(byte b31ToB24, byte b23ToB16, byte b15ToB8, byte b7ToB0) {
		int b31B24 = (b31ToB24 & 0xFF) <<24;			//On crée des entiers en 32bits à partir de nos bytes
		int b23B16 = (b23ToB16 & 0xFF) << 16;			//On les décale dans la chaine pour pouvoir ensuite les assembler
		int b15B8 = (b15ToB8 & 0xFF) << 8;				
		int b7B0 = (b7ToB0 & 0xFF);
		
		int extracted = b31B24|b23B16|b15B8|b7B0;		//On "assemble" nos entiers avec l'opérateur OU afin d'obtenir une chaîne de nos 32bits avec 4 bytes
		
		return extracted;
	}

	/**
	 * Parses an IDX file containing images
	 *
	 * @param data the binary content of the file
	 *
	 * @return A tensor of images
	 */
	public static byte[][][] parseIDXimages(byte[] data) {
		assert data.length > 15;//on vérifie qu'on appelle pas d'index out of range

		signConverter(data); // On transforme les pixels en pixels signés

		if (extractInt(data[0], data[1], data[2], data[3]) != 2051) {	//On vérifie qu'on ait bien un set d'images avec le nombre magique
			System.out.println("Vos données d'images ne correspondent pas à un set IDX");	//Si ce n'est pas ne cas on prévient l'utilisateur
		}else {
			byte[][][] images = new byte[extractInt(data[4], data[5], data[6], data[7])][extractInt(data[8], data[9], data[10], data[11])][extractInt(data[12], data[13], data[14], data[15])];
				//On a crée un tenseur adapté
			for(int i = 0; i < images.length; ++i) {
				for (int j = 0; j < images[0].length; ++j) {
					for (int k = 0; k < images[0][0].length; ++k) {
						images[i][j][k] = data[16+k+j*images[0][0].length+i*images[0].length*images[0][0].length];//On remplit chaque pixel dans notre tenseur avec la chaine de bytes des données
					}//On commence par ajouter 16 car les 16 premiers bytes nous on servi à paramétrer le tableau
				}//On ajoute k, qui représente l'avancée dans la ligne j de l'image i
				//Lorsque k redevient nul, il faut conserver la valeur qu'il avait en fin de ligne, on ajoute donc j fois le nombre des colonnes de l'image i, ce qui équivaut à toutes les lignes qu'a compté k
				//Enfin, lorsqu'une image est remplie et que l'on passe à la suivante (i augmente de 1 et k et j redeviennent nuls), il faut conserver la valeur de l'indice la chaîne data auquel laquelle on était situé, c'est à dire ajouter i fois la taille d'une image
			}
			return images;
		}
		return null;//on renvoie null si le nombre magique ne correspond pas
	}

	/**
	 * Parses an idx images containing labels
	 *
	 * @param data the binary content of the file
	 *
	 * @return the parsed labels
	 */
	public static byte[] parseIDXlabels(byte[] data) {
		assert data.length > 8;//on vérifie qu'on appelle pas d'index out of range
		if (extractInt(data[0], data[1], data[2], data[3]) != 2049){			//On vérifie qu'on ait bien un set de labels avec le nombre magique
			System.out.println("Vos données d'étiquettes ne correspondent pas à un set IDX");	//Si ce n'est pas ne cas on prévient l'utilisateur
		}else {
			int extractedLength = extractInt(data[4], data[5], data[6], data[7]);
			byte[] labels = new byte[extractedLength];	//Sinon, on crée le tableau des étiquettes de la longueur indiquée par le fichier
			
			for(int i = 0; i < labels.length; ++i) {
				labels[i] = data[(i+8)];		//On rempli le tableau avec tous les bytes restants qui correspondent aux étiquettes
			}
			return labels;
		}
		return null;
	}

	/**
	 * @brief Computes the squared L2 distance of two images
	 * 
	 * @param a, b two images of same dimensions
	 * 
	 * @return the squared euclidean distance between the two images
	 */
	public static float squaredEuclideanDistance(byte[][] a, byte[][] b) {
		// TODO: Implémenter
		assert (a.length == b.length) && (a[0].length== b[0].length);//on vérifie que les deux arrays sont de la même dimension
		//même si a priori si cette méthode est appelée ils le sont

		float EuclideanDistance = 0f;
		for(int row =0; row < a.length; ++row){//on itère sur les colonnes
			for (int col = 0; col < a[0].length; ++col){//puis sur les lignes
				EuclideanDistance += Math.pow(a[row][col]-b[row][col],2);//on ajoute le carré de la distance pixel à pixel à la distance totale
			}
		}
		//TODO: Gérer l'overflow
		return EuclideanDistance;
	}

	/**
	 * @brief Computes the inverted similarity between 2 images.
	 * 
	 * @param a, b two images of same dimensions
	 * 
	 * @return the inverted similarity between the two images
	 */


	public static float invertedSimilarity(byte[][] a, byte[][] b) {
		// TODO: Gérer l'overflow
		assert (a.length == b.length) && (a[0].length== b[0].length);//on vérifie que les deux arrays sont de la même dimension
		//même si a priori si cette méthode est appelée ils le sont

		float invertedSimilarity = 0f;
		float covariance = 0f, sumB =0f, sumA=0f, meanA, meanB, squaredStandardDeviationA=0f, squaredStandardDeviationB= 0f;

		//On commence par calculer la moyenne qui est nécessaire pour la suite du calcul
		for(int row =0; row < a.length; ++row) {//on itère sur les colonnes
			for (int col = 0; col < a[0].length; ++col) {//puis sur les lignes
			 sumA += a[row][col] ;//on somme d'abord les colorations pour éviter de faire la division à chaque pixel
			 sumB += b[row][col] ;
			}
		}
		meanA = sumA /(a.length * a[0].length); //on calcule enfin la moyenne arithmétique de la coloration des pixels de chaque image
		meanB = sumB /(a.length * a[0].length);

		//on calcule ensuite les 3 valeurs dont nous avons besoin pour trouver la similarité inversée en itérant à nouveau sur les images
		for(int row =0; row < a.length; ++row) {				//on itère sur les colonnes
			for (int col = 0; col < a[0].length; ++col) {			//puis sur les lignes
				covariance += (a [row] [col] - meanA)*(b[row][col] - meanB);
				squaredStandardDeviationA+= Math.pow(a[row][col] - meanA,2);		//l'écart type est la racine du carré de l'écart à la moyenne
				squaredStandardDeviationB+= Math.pow(b[row][col] - meanB,2);
			}
		}
		if(squaredStandardDeviationA*squaredStandardDeviationB == 0 ){return 2;		//si le dénominateur de la similarité est nul on retourne 2
		}else{
			invertedSimilarity = 1 - covariance / (float)Math.sqrt(squaredStandardDeviationA * squaredStandardDeviationB);		//sinon on applique la formule
		}

		return invertedSimilarity;
	}

	/**
	 * @brief Quicksorts and returns the new indices of each value.
	 * 
	 * @param values the values whose indices have to be sorted in non decreasing
	 *               order
	 * 
	 * @return the array of sorted indices
	 * 
	 *         Example: values = quicksortIndices([3, 7, 0, 9]) gives [2, 0, 1, 3]
	 */
	public static int[] quicksortIndices(float[] values) {
		// TODO: Implémenter
		assert values != null;//on vérifie que le tableau n'est pas vide
		int[] indices = new int[values.length];
		for(int i =0; i<values.length; ++i){indices[i] = i;}// on crée et remplit le tableau des indices
		quicksortIndices(values, indices, 0, values.length-1);


		return indices;
	}

	/**
	 * @brief Sorts the provided values between two indices while applying the same
	 *        transformations to the array of indices
	 * 
	 * @param values  the values to sort
	 * @param indices the indices to sort according to the corresponding values
	 * @param         low, high are the **inclusive** bounds of the portion of array
	 *                to sort
	 */
	public static void quicksortIndices(float[] values, int[] indices, int low, int high) {
		// TODO: Implémenter

		int lowIndex = low;
		int highIndex = high;//on initialise la valeur minimale et la valeur maximale au premier et dernier index de la liste
		float pivot = values[lowIndex];// le pivot est la valeur associée à l'indice le plus petit

		//on implémente l'algorithme réccursif quicksort classique
		do{
			if (values[lowIndex]< pivot){
				++lowIndex;
			}else if(values[highIndex] > pivot){
				--highIndex;
			}
			else {
				swap(lowIndex, highIndex, values, indices);
				++lowIndex;
				--highIndex;
			}
		}while ((lowIndex <= highIndex));
		if (low < highIndex){//ne pas oublier de changer la borne de la condition sinon on a une reccursion infinie
			quicksortIndices(values, indices, low, highIndex);
		}
		if( high > lowIndex){
			quicksortIndices(values, indices, lowIndex, high);
		}
	}

	/**
	 * @brief Swaps the elements of the given arrays at the provided positions
	 * 
	 * @param         i, j the indices of the elements to swap
	 * @param values  the array floats whose values are to be swapped
	 * @param indices the array of ints whose values are to be swapped
	 */
	public static void swap(int i, int j, float[] values, int[] indices) {
		// TODO: Implémenter
		float tempValue;
		int tempIndex; // on initialise deux variables temporaires, une pour les indices et une pour les valeurs

		tempValue = values[i];//on donne la valeur des listes à l'index i aux valeurs temporaires
		tempIndex = indices[i];

		values[i]= values[j];//on assigne la valeur des listes à l'index j à celles de l'index i
		indices[i] = indices[j];

		values[j]= tempValue;// et vice versa à l'aide de la valeur temporaire
		indices[j] = tempIndex;


	}

	/**
	 * @brief Returns the index of the largest element in the array
	 * 
	 * @param array an array of integers
	 * 
	 * @return the index of the largest integer
	 */
	public static int indexOfMax(int[] array) {
		// TODO: Implémenter
		assert array != null;
		int max = array[0];
		int indexMax=0;
		for(int index =0; index < array.length; index++){
			if (array[index] > max) {
				max = array[index];
				indexMax = index;
			}
		}
		return indexMax;
	}

	/**
	 * The k first elements of the provided array vote for a label
	 *
	 * @param sortedIndices the indices sorted by non-decreasing distance
	 * @param labels        the labels corresponding to the indices
	 * @param k             the number of labels asked to vote
	 *
	 * @return the winner of the election
	 */
	public static byte electLabel(int[] sortedIndices, byte[] labels, int k) {
		// TODO: Implémenter
		int[]votes = new int[10];
		for(int i= 0; i< k ; ++i){
			votes[labels[sortedIndices[i]]] += 1;
		}

		return (byte)indexOfMax(votes);
	}

	/**
	 * Classifies the symbol drawn on the provided image
	 *
	 * @param image       the image to classify
	 * @param trainImages the tensor of training images
	 * @param trainLabels the list of labels corresponding to the training images
	 * @param k           the number of voters in the election process
	 *
	 * @return the label of the image
	 */
	public static byte knnClassify(byte[][] image, byte[][][] trainImages, byte[] trainLabels, int k) {
		// TODO: Implémenter
		assert image != null && trainImages != null;

		float distances[] = new float[trainImages.length];

		for(int i = 0; i<trainImages.length ; ++i){
			distances[i]= invertedSimilarity(image,trainImages[i]);
		}

		int[] closestToFurthest = quicksortIndices(distances);

		return  electLabel(closestToFurthest, trainLabels, k);
	}

	/**
	 * Computes accuracy between two arrays of predictions
	 * 
	 * @param predictedLabels the array of labels predicted by the algorithm
	 * @param trueLabels      the array of true labels
	 * 
	 * @return the accuracy of the predictions. Its value is in [0, 1]
	 */
	public static double accuracy(byte[] predictedLabels, byte[] trueLabels) {
		// TODO: Implémenter
		int numberOfRightPredictions=0;
		double accuracy = 0.;

		for (int i =0; i<predictedLabels.length; ++i) {
			if (predictedLabels[i] == trueLabels[i]){
				++numberOfRightPredictions ;
			}
		}
		return (double)numberOfRightPredictions / predictedLabels.length *100;
	}
}
