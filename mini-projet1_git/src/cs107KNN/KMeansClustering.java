package cs107KNN;

import java.util.*;

public class KMeansClustering {
	public static void main(String[] args) {
		int K =10;
		int maxIters = 35;

		// TODO: Adaptez les parcours

		byte[][][] images = KNN.parseIDXimages(Helpers.readBinaryFile("datasets/10-per-digit_images_train"));
		byte[] labels = KNN.parseIDXlabels(Helpers.readBinaryFile("datasets/10-per-digit_labels_train"));

		byte[][][] reducedImages = KMeansReduce(images, K, maxIters);

		byte[] reducedLabels = new byte[reducedImages.length];
		for (int i = 0; i < reducedLabels.length; i++) {
			reducedLabels[i] = KNN.knnClassify(reducedImages[i], images, labels, 5);
			System.out.println("Classified " + (i + 1) + " / " + reducedImages.length);
		}

		Helpers.writeBinaryFile("datasets/xptdr_images", encodeIDXimages(reducedImages));
		Helpers.writeBinaryFile("datasets/xptdr_labels", encodeIDXlabels(reducedLabels));

		int TESTS = 700;
		int K2 =1;
		byte[][][] trainImages =
				KNN.parseIDXimages(Helpers.readBinaryFile("datasets/reduced10Kto1K_images_fictive"));
		byte[] trainLabels =
				KNN.parseIDXlabels(Helpers.readBinaryFile("datasets/reduced10Kto1K_labels_fictive"));

		byte[][][] testImages = KNN.parseIDXimages(Helpers.readBinaryFile("datasets/10k_images_test"));

		byte[] testLabels = KNN.parseIDXlabels(Helpers.readBinaryFile("datasets/10k_labels_test"));

		byte[] predictions = new byte[TESTS];

		long start = System.currentTimeMillis();

		for (int i = 0; i < TESTS; i++) {
			predictions[i] = KNN.knnClassify(testImages[i], trainImages, trainLabels, K2);
		}
		long end = System.currentTimeMillis();
		Helpers.show("Test", testImages, predictions, testLabels, 20, 35) ;

		System.out.println("Taux de précision : "+KNN.accuracy(predictions, testLabels)+"%");



		double time = (end - start) / 1000d;

		System.out.println("temps de calcul : " + time);
		System.out.println("Temps par image : "+ time/TESTS);
	}


    /**
     * @brief Encodes a tensor of images into an array of data ready to be written on a file
     * 
     * @param images the tensor of image to encode
     * 
     * @return the array of byte ready to be written to an IDX file
     */
	public static byte[] encodeIDXimages(byte[][][] images) {
		// TODO: Implémenter
		 byte[] encodedIdxImages = new byte[16+images.length*images[0].length*images[0][0].length];

		 encodeInt(2051, encodedIdxImages, 0);
		 encodeInt(images.length, encodedIdxImages, 4);
		 encodeInt(images[0].length, encodedIdxImages, 8);
		 encodeInt(images[0][0].length, encodedIdxImages, 12);


		for(int i = 0; i < images.length; ++i) {
			for (int j = 0; j < images[0].length; ++j) {
				for (int k = 0; k < images[0][0].length; ++k) {
				encodedIdxImages[16 + k + j*images[0][0].length + i*images[0].length*images[0][0].length] = (byte)((images[i][j][k]+128) & 0xFF);
				}
			}
		}

		return encodedIdxImages;
	}

    /**
     * @brief Prepares the array of labels to be written on a binary file
     * 
     * @param labels the array of labels to encode
     * 
     * @return the array of bytes ready to be written to an IDX file
     */
	public static byte[] encodeIDXlabels(byte[] labels) {
		// TODO: Implémenter
		byte[]encodedIDXLabel = new byte[labels.length + 8];

		encodeInt(2049,encodedIDXLabel,0);//on remet le nombre magique
		encodeInt(labels.length,encodedIDXLabel,4);

		for(int i = 8; i < encodedIDXLabel.length ;++i){
			encodedIDXLabel[i]= labels[i-8];
		}

		return encodedIDXLabel;
	}

    /**
     * @brief Decomposes an integer into 4 bytes stored consecutively in the destination
     * array starting at position offset
     * 
     * @param n the integer number to encode
     * @param destination the array where to write the encoded int
     * @param offset the position where to store the most significant byte of the integer,
     * the others will follow at offset + 1, offset + 2, offset + 3
     */
	public static void encodeInt(int n, byte[] destination, int offset) {
		assert destination.length > offset+3;
		// TODO: Implémenter
		byte b7To0 = (byte) (n & 0xFF);
		byte b15to8 = (byte) ((n & 0xFF << 8)>>8) ;
		byte b23To16 = (byte) ((n & 0xFF << 16 )>>16) ;
		byte b31To24 = (byte)((n & 0xFF << 24)>>24);

		destination[offset] = b31To24;
		destination[offset+1] = b23To16;
		destination[offset+2] = b15to8;
		destination[offset+3] = b7To0;

	}

    /**
     * @brief Runs the KMeans algorithm on the provided tensor to return size elements.
     * 
     * @param tensor the tensor of images to reduce
     * @param size the number of images in the reduced dataset
     * @param maxIters the number of iterations of the KMeans algorithm to perform
     * 
     * @return the tensor containing the reduced dataset
     */
	public static byte[][][] KMeansReduce(byte[][][] tensor, int size, int maxIters) {
		assert tensor != null && tensor[0] != null ;
		int[] assignments = new Random().ints(tensor.length, 0, size).toArray();
		System.out.println(Arrays.toString(assignments));
		byte[][][] centroids = new byte[size][][];
		initialize(tensor, assignments, centroids);
		System.out.println(Arrays.toString(assignments));
		int nIter = 0;

		while (nIter < maxIters) {
			byte[][][] previousCentroids = centroids.clone();

			// Step 1: Assign points to closest centroid
			recomputeAssignments(tensor, centroids, assignments);
			System.out.println(Arrays.toString(assignments));
			System.out.println("Recomputed assignments");
			// Step 2: Recompute centroids as average of points
			recomputeCentroids(tensor, centroids, assignments);

			System.out.println("Recomputed centroids");

			System.out.println("KMeans completed iteration " + (nIter + 1) + " / " + maxIters);

			if(previousCentroids == centroids){break;}
			++nIter;
		}

		return centroids;
	}

   /**
     * @brief Assigns each image to the cluster whose centroid is the closest.
     * It modifies.
     * 
     * @param tensor the tensor of images to cluster
     * @param centroids the tensor of centroids that represent the cluster of images
     * @param assignments the vector indicating to what cluster each image belongs to.
     *  if j is at position i, then image i belongs to cluster j
     */
	public static void recomputeAssignments(byte[][][] tensor, byte[][][] centroids, int[] assignments) {

		assert centroids != null && centroids.length !=0 && tensor != null && assignments != null;

		for (int i=0; i < tensor.length; ++i){

			float[] distancesToCentroids = new float[centroids.length];

			for(int j= 0; j < centroids.length; ++j){
				distancesToCentroids[j]= KNN.squaredEuclideanDistance(tensor[i],centroids[j]);
			}
			assignments[i] = KNN.quicksortIndices(distancesToCentroids)[0];
		}

	}



	/**
     * @brief Computes the centroid of each cluster by averaging the images in the cluster
     * 
     * @param tensor the tensor of images to cluster
     * @param centroids the tensor of centroids that represent the cluster of images
     * @param assignments the vector indicating to what cluster each image belongs to.
     *  if j is at position i, then image i belongs to cluster j
     */

    public static void recomputeCentroids(byte[][][] tensor, byte[][][] centroids, int[] assignments) {

    	assert centroids != null && centroids.length !=0 && tensor != null && assignments != null;

		ArrayList <byte[][]> [] clusters = new ArrayList[centroids.length];


		//on initialise les Arraylists sinon ça ne marche pas et on a des nullpointers exceptions
		for(int cluster =0; cluster< clusters.length; cluster++){clusters[cluster]= new ArrayList<>();}

		for(int i = 0; i< tensor.length;++i){
			clusters[assignments[i]].add(tensor[i]);
		}
		//byte[][][] fictiveCentroids = new byte[clusters.length][tensor[0].length][tensor[0][0].length];



		for(int cluster = 0; cluster < clusters.length;++cluster ){
			if (clusters[cluster].size()==0){break;}


			//calcule la moyenne dans chaque cluster


			for (int j =0; j< centroids[0].length; j++){

				for (int k = 0; k < centroids[0][0].length;++k){

					double sumOfPixels = 0f;

					for (int l = 0; l < clusters[cluster].size(); ++l){

						sumOfPixels += clusters[cluster].get(l)[j][k] ;
						//on crée ici un centroid fictif du cluster qu'on utilise ensuite pour trouver le centroid
						centroids[cluster][j][k] = (byte) (sumOfPixels / (double)clusters[cluster].size());

					}
				}
			}

			//après avoir l'image moyenne du cluster on calcule la distance de chaque image à ce cluster et on désigne l'image la plus proche de cette image le nouvea centroid

			/** float [] distancesCluster = new float[clusters[cluster].size()];
			for (int l = 0; l < clusters[cluster].size(); ++l){
				distancesCluster[l] = KNN.squaredEuclideanDistance(fictiveCentroids[cluster],clusters[cluster].get(l));
			}

			centroids[cluster] = clusters[cluster].get(KNN.quicksortIndices(distancesCluster)[0]); **/

		}

	}

    /**
     * Initializes the centroids and assignments for the algorithm.
     * The assignments are initialized randomly and the centroids
     * are initialized by randomly choosing images in the tensor.
     * 
     * @param tensor the tensor of images to cluster
     * @param assignments the vector indicating to what cluster each image belongs to.
     * @param centroids the tensor of centroids that represent the cluster of images
     *  if j is at position i, then image i belongs to cluster j
     */
	public static void initialize(byte[][][] tensor, int[] assignments, byte[][][] centroids) {
		Set <Integer> centroidIds = new HashSet<>();
		Random r = new Random("cs107-2018".hashCode());
		while (centroidIds.size() != centroids.length)
			centroidIds.add(r.nextInt(tensor.length));
		Integer[] cids = centroidIds.toArray(new Integer[] {});
		for (int i = 0; i < centroids.length; i++)
			centroids[i] = tensor[cids[i]];
		for (int i = 0; i < assignments.length; i++)
			assignments[i] =   cids[r.nextInt(cids.length)];
	}
}
