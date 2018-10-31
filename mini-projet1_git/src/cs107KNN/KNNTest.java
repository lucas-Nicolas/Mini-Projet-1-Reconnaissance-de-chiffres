package cs107KNN;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class KNNTest {
	public static void main(String[] args) {
		// TODO: Adapt path to data files in parsing test
		// Decommentez au fur et à mesure que vous implémentez
		extractIntTest();
		parsingTest();
		squaredEuclideanDistanceTest();
		invertedSimilarityTest();
		quicksortTest();
		quickSortAdvancedTest(10000);
		indexOfMaxTest();
		electLabelTest();
		knnClassifyTest();
		accuracyTest();

		showDataset("datasets/reduced10Kto1K_images_fictive","datasets/reduced10Kto1K_labels_fictive");
	}

	public static void extractIntTest() {
		byte b1 = 40; // 00101000
		byte b2 = 120; // 00010100
		byte b3 = 70; // 00001010
		byte b4 = -117; // 00000101

		String expected = Helpers.byteToBinaryString(b1) +
			Helpers.byteToBinaryString(b2) +
			Helpers.byteToBinaryString(b3) +
			Helpers.byteToBinaryString(b4);

		int obtained = KNN.extractInt(b1, b2, b3, b4);
		int expectedInt = (b1 & 0xFF) <<24  | (b2 & 0xFF) << 16 |(b3 & 0xFF) <<8|  (b4 & 0xFF) ;
		System.out.println("=== Test extractInt ===");
		System.out.println("Entier attendu:\t " + expected+ "   en entier : " + expectedInt);
		System.out.println("extractInt produit:\t " + obtained);	//Test ok
	}

	public static void parsingTest() {
		System.out.println("=== Test parsing ===");

		byte[][][] images = KNN.parseIDXimages(Helpers.readBinaryFile("datasets/10-per-digit_images_train"));
		byte[] labels = KNN.parseIDXlabels(Helpers.readBinaryFile("datasets/10-per-digit_labels_train"));

		System.out.println("Number of images: " + images.length);
		System.out.println("Height: " + images[0].length);
		System.out.println("Width: " + images[0][0].length);

		Helpers.show("Test parsing", images, labels, 10, 10);
	}


	public static void squaredEuclideanDistanceTest() {
		System.out.println("=== Test distance euclidienne ===");
		byte[][] a = new byte[][] {{1, 1}, {2, 2}};
		byte[][] b = new byte[][] {{3, 3}, {4, 4}};

		System.out.println("Distance calculée: " + KNN.squaredEuclideanDistance(a, b));
		System.out.println("Distance attendue: 16.0");
	}

	public static void invertedSimilarityTest() {
		System.out.println("=== Test similarité inversée ===");
		byte[][] a = new byte[][] {{1, 1}, {1, 2}};
		byte[][] b = new byte[][] {{50, 50}, {50, 100}};

		System.out.println("Distance calculée: " + KNN.invertedSimilarity(a, b));
		System.out.println("Distance attendue: 0.0");
	}

	public static void quicksortTest() {
		System.out.println("=== Test quicksort ===");
		float[] data = new float[] {0,1,2,3,4,1.5f,2.5f,7,0.4f};
		int[] result = KNN.quicksortIndices(data);

		System.out.println("Indices triés: " + Arrays.toString(result));
	}

	public static void indexOfMaxTest() {
		System.out.println("=== Test indexOfMax ===");
		int[] data = new int[]{0, 5, 9, 1};

		int indexOfMax = KNN.indexOfMax(data);
		System.out.println("Indices: [0, 1, 2, 3]");
		System.out.println("Données: " + Arrays.toString(data));
		System.out.println("L'indice de l'élément maximal est: " + indexOfMax);
	}


	public static void electLabelTest() {
		System.out.println("=== Test electLabel ===");
		int[] sortedIndices = new int[]{0, 3, 2, 1};
		byte[] labels = new byte[]{2, 1, 1, 2};
		int k = 3;

		System.out.println("Étiquette votée: " + KNN.electLabel(sortedIndices, labels, k));
		System.out.println("Étiquette attendue: 2");
	}

	public static void knnClassifyTest() {
		System.out.println("=== Test predictions ===");
		byte[][][] imagesTrain = KNN.parseIDXimages(Helpers.readBinaryFile("datasets/10-per-digit_images_train"));
		byte[] labelsTrain = KNN.parseIDXlabels(Helpers.readBinaryFile("datasets/10-per-digit_labels_train"));

		byte[][][] imagesTest = KNN.parseIDXimages(Helpers.readBinaryFile("datasets/10k_images_test"));
		byte[] labelsTest = KNN.parseIDXlabels(Helpers.readBinaryFile("datasets/10k_labels_test"));

		byte[] predictions = new byte[60];
		for (int i = 0; i < 60; i++) {
			predictions[i] = KNN.knnClassify(imagesTest[i], imagesTrain, labelsTrain, 7);
		}
		Helpers.show("Test predictions", imagesTest, predictions, labelsTest, 10, 6);
	}


	public static void accuracyTest() {
		System.out.println("=== Test précision ===");
		byte[] a = new byte[] {1, 1, 1, 1};
		byte[] b = new byte[] {1, 1, 1, 9};


		System.out.println("Précision calculée: " + KNN.accuracy(a, b));
		System.out.println("Précision attendue:  0.75");
	}


	public static void quickSortAdvancedTest (int length) {

		float[] testArray = new float[length];
		System.out.println("=== Test Quicksort avancé === ");

		for(int i = 0;i<length; ++i){
			testArray[i] = (float) Math.random() * 100 ;

		}
		System.out.println("liste non triée : " + Arrays.toString(testArray));

		ArrayList <Float> testArrayCopy  = new ArrayList<>();

		for(int i = 0;i<length; ++i){

			testArrayCopy.add(testArray[i]);

		}


		int[] indices = KNN.quicksortIndices(testArray);
		int indicesTest [] = new int[length];
		for(int i = 0;i<length; ++i){
			indicesTest[i] = testArrayCopy.indexOf(testArray[i]);
		}






		Collections.sort(testArrayCopy);

		System.out.println("liste triée : " + testArrayCopy.toString());
		System.out.println("liste triée : " + Arrays.toString(testArray)+"\n"+"\n");
		System.out.println("Indices attendus"+ Arrays.toString(indicesTest));
		System.out.println ( "Indices triés : " +Arrays.toString(indices));
	}




	public static void showDataset(String filename,String labelsName){
		byte[][][] trainImages =
				KNN.parseIDXimages(Helpers.readBinaryFile(filename));
		byte[]trainLabels= KNN.parseIDXlabels(Helpers.readBinaryFile(labelsName));
		Helpers.show("lol",trainImages,trainLabels,10,10);
	}
}
