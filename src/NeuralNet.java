import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;

public class NeuralNet {
	
	public static void main(String args[])
	{
		String base = "/Users/lherbeur/weka-3-8-3/data/";
		String [] filePaths = {"breast-cancer.arff", "diabetes.arff", "soybean.arff", "vote.arff"};
		
		for (String s: filePaths)
		{
			simpleWekaTrain(base + s);
		}
	}
	
	
	public static void simpleWekaTrain(String filepath)
	{
		try{
		FileReader trainreader = new FileReader(filepath);
		System.out.println(filepath);
		
		//split into train and test
		Instances train = new Instances(trainreader);
		train.setClassIndex(train.numAttributes()-1);
		
		//Instance of NN
		MultilayerPerceptron mlp = new MultilayerPerceptron();
		//Setting Parameters
		mlp.setLearningRate(0.1);
		mlp.setMomentum(0.2);
		mlp.setTrainingTime(2000);
		//shallow
		mlp.setHiddenLayers("1");
		//deep
	//	mlp.setHiddenLayers("6");
		mlp.buildClassifier(train);
		
		//evaluate 
		Evaluation eval = new Evaluation(train);
		
		//cross-valdtn
	    eval.crossValidateModel(mlp, train, 10, new Random(1));
	    
	    eval.evaluateModel(mlp, train);
	    System.out.println("Error rate - "+eval.errorRate()); 
	    System.out.println(eval.toSummaryString()); 
	    
		
		}
		catch(Exception ex){
		ex.printStackTrace();
		}
	}
	
//	static ArrayList <ArrayList<Instance>>splitData(Instances train)
//	{
//		int numTrain = (int)(0.75 * train.size());
//		int numTest = train.size() - numTrain;
//				
//		ArrayList <Instance> trainData = new ArrayList();
//		ArrayList <Instance> testData = new ArrayList();  
//		ArrayList <ArrayList<Instance>> data = new ArrayList();  
//		
//		
//		for (Instance i: train)
//		{
//			double rand = Math.random();
//			if (rand <= 0.75)
//			{
//				trainData.add(i);
//			}
//			else
//				testData.add(i);
//			
//		}
//		
//		data.add(trainData);
//		data.add(testData);
//		
//		return data;
//	}
	
//	public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
//	    Instances[][] split = new Instances[2][numberOfFolds];
//
//	    for (int i = 0; i < numberOfFolds; i++) {
//	        split[0][i] = data.trainCV(numberOfFolds, i);
//	        split[1][i] = data.testCV(numberOfFolds, i);
//	    }        
//	    return split;
//	}//corssValidationSplit().


}
