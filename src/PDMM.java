/**
 * 
 */

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import java.util.Map.Entry;

public class PDMM {
	
	public Set<String> wordSet;
	public int numTopic;
	public double alpha, beta;
	public double namda; // the parameter of Poisson distribution
	public int numIter;
	public int saveStep;
	public ArrayList<Document> docList;
	public int roundIndex;
	private Random rg;
	public double threshold; // the similar threshold for constructing similar word set
	public double weight; //the promotion of similar word
	public int topWords;
	public int filterSize; // the filter size for filtering similar word set
	public String word2idFileName;
	public String similarityFileName;
	public boolean use_gpu; //true for GPU-PDMM, false for PDMM

	public Map<String, Integer> word2id;
	public Map<Integer, String> id2word;
	public Map<Integer, Double> wordIDFMap;
	
	public Map<Integer, Set<Integer>> ZdMap; 
	public int[] TdArray;
	
	public Map<Integer, Map<Integer, Double>> docUsefulWords;
	public ArrayList<ArrayList<Integer>> topWordIDList;
	public int vocSize;
	public int numDoc;
	public int maxTd; // the maximum number of topics within a document
	private double[][] schema;
	public Map<Integer, int[]> docToWordIDListMap;
	public double[][] phi;
	private double[] pz;
	private double[][] pdz;
	private double[][] topicProbabilityGivenWord;
	private double[][] pwz;

	public ArrayList<ArrayList<Boolean>> wordGPUFlag; // wordGPUFlag.get(doc).get(wordIndex)
	public Map<Integer, int[]> assignmentListMap; // topic assignment for every document
	public ArrayList<ArrayList<Map<Integer, Double>>> wordGPUInfo;

	private double[] nz; // [topic]; nums of words in every topic
	private double[][] nzw; // V_{.k}
	
	private int[] Ck; // Ck[topic]
	private int CkSum;
	public int searchTopK; // search size for heuristic search , 
	
	private Map<Integer, Map<Integer, Double>> schemaMap;
	
	public PDMM(ArrayList<Document> doc_list, int num_topic, int num_iter, int save_step, double beta,
			double alpha, double namda, double threshold) {
		docList = doc_list;
		numDoc = docList.size();
		numTopic = num_topic;
		this.alpha = alpha;
		numIter = num_iter;
		saveStep = save_step;
		this.threshold = threshold;
		this.beta = beta;
		this.namda = namda;
	}
	
	public boolean loadWordMap(String filename) {
		try {
			FileInputStream fis = new FileInputStream(filename);
			InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
			BufferedReader reader = new BufferedReader(isr);
			String line;
			
			//construct word2id map
			//the format of one line of word2id: "apple,0'
			while ((line = reader.readLine()) != null) {
				line = line.trim();
				String[] items = line.split(",");
				word2id.put(items[0], Integer.parseInt(items[1]));
				id2word.put(Integer.parseInt(items[1]), items[0]);
			}
			System.out.println("finish read wordmap and the num of word is " + word2id.size());
			return true;
		} catch (Exception e) {
			System.out.println("Error while reading other file:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
	}
	
	/**
	 * Collect the similar words Map, not including the word itself
	 * 
	 * @param filename:
	 *            shcema_similarity filename
	 * @param threshold:
	 *            if the similarity is bigger than threshold, we consider it as
	 *            similar words
	 * @return
	 */
	public Map<Integer, Map<Integer, Double>> loadSchema(
			String filename, double threshold) {
		int word_size = word2id.size();
		Map<Integer, Map<Integer, Double>> schemaMap = new HashMap<Integer, Map<Integer, Double>>();
		try {
			FileInputStream fis = new FileInputStream(filename);
			InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
			BufferedReader reader = new BufferedReader(isr);
			String line;
			int lineIndex = 0;
			while ((line = reader.readLine()) != null) {
				line = line.trim();
				String[] items = line.split(" ");

				for (int i = 0, length = items.length; i < length; i++) {
					double value = Double.parseDouble(items[i]);
					schema[lineIndex][i] = value;
				}
				
				lineIndex++;
				if (lineIndex % 500 == 0) {
					System.out.println(lineIndex);
				}
			}
			
			double count = 0.0;
			Map<Integer, Double> tmpMap = null;
			for (int i = 0; i < word_size; i++) {
				tmpMap = new HashMap<Integer, Double>();
				double v = 0;
				for (int j = 0; j < word_size; j++) {
					// we need to calculate the similar word set for each word i
					// so we utilize the i-th row of similarity matrix 
					// so [j][i] is used below
					v = schema[j][i];
					if (Double.compare(v, threshold) > 0) {
						// whether the word belongs to similar word
						tmpMap.put(j, v);
					}
				}
				if (tmpMap.size() > filterSize) {
					// large similar word set means that the word is general, so discard it
					tmpMap.clear();
				}
				tmpMap.remove(i);
				if (tmpMap.size() == 0) {
					continue;
				}
				count += tmpMap.size();
				schemaMap.put(i, tmpMap);
			}
			schema = null;
			System.out.println("finish read schema, the avrage number of value is " + count / schemaMap.size());
			return schemaMap;
		} catch (Exception e) {
			System.out.println("Error while reading other file:" + e.getMessage());
			e.printStackTrace();
			return null;
		}
	}
	
	/**
	 * 
	 * @param wordID
	 * @param topic
	 * @return word probability given topic 
	 */
	public double getWordProbabilityUnderTopic(int wordID, int topic) {
		return (nzw[topic][wordID] + beta) / (nz[topic] + vocSize * beta);
	}
	
	public double getMaxTopicProbabilityForWord(int wordID) {
		double max = -1.0;
		double tmp = -1.0;
		for (int t = 0; t < numTopic; t++) {
			tmp = getWordProbabilityUnderTopic(wordID, t);
			if (Double.compare(tmp, max) > 0) {
				max = tmp;
			}
		}
		return max;
	}
	
	public double findTopicMaxProbabilityGivenWord(int wordID) {
		double max = -1.0;
		double tmp = -1.0;
		for (int i = 0; i < numTopic; i++) {
			tmp = topicProbabilityGivenWord[wordID][i];
			if (Double.compare(tmp, max) > 0) {
				max = tmp;
			}
		}
		return max;
	}

	public double getTopicProbabilityGivenWord(int topic, int wordID) {
		return topicProbabilityGivenWord[wordID][topic];
	}
	
	/**
	 * update GPU flag, which decide whether do GPU operation or not
	 * @param docID
	 * @param newTopic
	 */
	public void updateWordGPUFlag(int docID, int word, int index, int newTopic) {
		// we calculate the p(t|w) and p_max(t|w) and use the ratio to decide we
		// use gpu for the word under this topic or not
		// see word filtering part in our paper
		double maxProbability = findTopicMaxProbabilityGivenWord(word);
		double ratio = getTopicProbabilityGivenWord(newTopic, word) / maxProbability;
		double a = rg.nextDouble();
		if(Double.compare(ratio, a) > 0){
			wordGPUFlag.get(docID).set(index, true);
		}
		else{
			wordGPUFlag.get(docID).set(index, false);
		}
		
	}

	/**
	 * load initial status from extra file if you need it
	 * we don't use that, random initialization is used in our experiment setting
	 * @param filename for topic assignment for each document
	 */
	public void loadInitialStatus(String filename) {
		try {
			FileInputStream fis = new FileInputStream(filename);
			InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
			BufferedReader reader = new BufferedReader(isr);
			String line;
			Set<Integer> Zd = null;
			while ((line = reader.readLine()) != null) {
				line = line.trim();
				String[] items = line.split(" ");
				assert(items.length == numDoc);
				for (int i = 0, length_items = items.length; i < length_items; i++) {
					int topic = Integer.parseInt(items[i]);
					Zd = new HashSet<Integer>();
					Zd.add(topic);
					ZdMap.put(i, Zd);
					TdArray[i] = 1;
					normalCountZd(Zd, +1);
					int[] termIDArray = docToWordIDListMap.get(i);
					for (int j = 0, num_word = termIDArray.length; j <num_word; j++){
						assignmentListMap.get(i)[j] = topic;
						int word = termIDArray[j];
						normalCountWord(topic, word, +1);	
					}
				}
				break;
			}

			System.out.println("finish loading initial status");
		} catch (Exception e) {
			System.out.println("Error while reading other file:" + e.getMessage());
			e.printStackTrace();
		}
	}

	public void ratioCount(Integer topic, Integer docID, int word, int index, int flag) {
		nzw[topic][word] += flag;
		nz[topic] += flag;
		
		// we update gpu flag for every document before it change the counter
		// when adding numbers
		if (flag > 0) {
			updateWordGPUFlag(docID, word, index, topic);
			boolean gpuFlag = wordGPUFlag.get(docID).get(index);
			Map<Integer, Double> gpuInfo = new HashMap<Integer, Double>();
			if (gpuFlag) { // do gpu count
				if (schemaMap.containsKey(word)) {
					Map<Integer, Double> valueMap = schemaMap.get(word);
						// update the counter
					for (Map.Entry<Integer, Double> entry : valueMap.entrySet()) {
						int k = entry.getKey();
						double v = weight; // the promotion of similar word
						nzw[topic][k] += v;
						nz[topic] += v;
						gpuInfo.put(k, v);
					} // end loop for similar words
			//		valueMap.clear();
				} else { // schemaMap don't contain the word
					// the word doesn't have similar words, the infoMap is empty
					// we do nothing
					}
				} else { // the gpuFlag is False
					// it means we don't use gpu, so the gouInfo map is empty
				}
				wordGPUInfo.get(docID).set(index, gpuInfo); // we update the gpuinfo
														// map
		} else { // we do subtraction according to last iteration information
				Map<Integer, Double> gpuInfo = wordGPUInfo.get(docID).get(index);
				if (gpuInfo.size() > 0) {
					for (int similarWordID : gpuInfo.keySet()) {
						double v = weight;
						nzw[topic][similarWordID] -= v;
						nz[topic] -= v;
					}
				}
			}
		}

	
	// below functions are used for updating the counter for Gibbs sampling
	public void normalCountWord(Integer topic, int word, Integer flag) {
		nzw[topic][word] += flag;
		nz[topic] += flag;
	}
	

	public void normalCountZd(Set<Integer> Zd, Integer flag){
		for (int topic : Zd){
			Ck[topic] += flag;
			CkSum += flag;
		}
	}
	
	public Set<Integer> getdk_Zd(
			int docID, int[] assignment, int topic){
		Set<Integer> dk = new HashSet<Integer>();
		int[] termIDArray = docToWordIDListMap.get(docID);
		for(int i = 0, length = assignment.length; i < length; i++){
			int z = assignment[i];
			if (z==topic){
				dk.add(termIDArray[i]);
			}
		}
		return dk;
	}
	
	public Map<Integer, Map<Integer, Integer>> getNdkt_Zd(
			int docID, int[] ZdList, int[] assignment){
		Map<Integer, Map<Integer, Integer>> Ndkt = new HashMap<Integer,Map<Integer, Integer>>();
		for(int k : ZdList){
			Ndkt.put(k, new HashMap<Integer, Integer>());
		}
		int[] termIDArray = docToWordIDListMap.get(docID);
		for(int i = 0, length = termIDArray.length; i < length; i++){
			int word = termIDArray[i];
			int topic = assignment[i];
			if (Ndkt.get(topic).containsKey(word)){
				Ndkt.get(topic).put(word, Ndkt.get(topic).get(word)+1);
			}
			else{
				Ndkt.get(topic).put(word, 1);
			}
		}
		return Ndkt;
	}
	
	public Map<Integer, Integer> getNdk_Zd(
			int docID, int[] ZdList, int[] assignment){
		 Map<Integer, Integer> Ndk = new HashMap<Integer,Integer>();
		for(int k : ZdList){
			Ndk.put(k,0);
		}
		int[] termIDArray = docToWordIDListMap.get(docID);
		for(int i = 0, length = termIDArray.length; i < length; i++){
			int word = termIDArray[i];
			int topic = assignment[i];
			Ndk.put(topic, Ndk.get(topic)+1);
		}
		return Ndk;
	}
	
	public void initNewModel() {
		wordGPUFlag = new ArrayList<ArrayList<Boolean>>();
		docToWordIDListMap = new HashMap<Integer, int[]>();
		word2id = new HashMap<String, Integer>();
		id2word = new HashMap<Integer, String>();
		wordIDFMap = new HashMap<Integer, Double>();
		docUsefulWords = new HashMap<Integer, Map<Integer, Double>>();
		wordSet = new HashSet<String>();
		topWordIDList = new ArrayList<ArrayList<Integer>>();
		assignmentListMap = new HashMap<Integer, int[]>();
		wordGPUInfo = new ArrayList<ArrayList<Map<Integer, Double>>>();
		rg = new Random();
		
		ZdMap = new HashMap<Integer, Set<Integer>>(); 
		TdArray = new int[docList.size()];
		// construct vocabulary
		loadWordMap(word2idFileName);

		vocSize = word2id.size();
		phi = new double[numTopic][vocSize];
		pz = new double[numTopic];
		pdz = new double[numDoc][numTopic];
		pwz = new double[vocSize][numTopic];

		schema = new double[vocSize][vocSize];
		topicProbabilityGivenWord = new double[vocSize][numTopic];
		
		Document doc = null;
		ArrayList<Boolean> docWordGPUFlag = null;
		ArrayList<Map<Integer, Double>> docWordGPUInfo = null;
		int[] termIDArray = null;
		
		for (int i = 0; i < numDoc; i++) {
			doc = docList.get(i);
			assignmentListMap.put(i, new int[doc.words.length]);
			termIDArray = new int[doc.words.length];
			docWordGPUFlag = new ArrayList<Boolean>();
			docWordGPUInfo = 
				new ArrayList<Map<Integer, Double>>();
			for (int j = 0, num_word = doc.words.length; j < num_word; j++) {
				termIDArray[j] = word2id.get(doc.words[j]);
				docWordGPUFlag.add(false); // initial for False for every word
				docWordGPUInfo.add(new HashMap<Integer, Double>());
			}
			wordGPUFlag.add(docWordGPUFlag);
			wordGPUInfo.add(docWordGPUInfo);
			docToWordIDListMap.put(i, termIDArray);
		}

		// init the counter
		nz = new double[numTopic];
		nzw = new double[numTopic][vocSize];
		Ck = new int[numTopic];
		CkSum = 0;
	}
	
	
	// initialize the model
	public void init_GSDMM() {
		schemaMap = loadSchema(similarityFileName, threshold);

		double[] ptd = new double[maxTd];
		double temp_factorial = 1.0;
		for ( int i = 0; i < maxTd; i++ ){
			temp_factorial *= (i+1);
			ptd[i] = Math.pow(namda, (double)(i+1)) * Math.exp(-namda)/temp_factorial;
		}
		
		for (int i = 1; i < maxTd; i++) {
			ptd[i] += ptd[i - 1];
		}
		
		for (int d = 0; d < numDoc; d++) {
			
			double u = rg.nextDouble() * ptd[ptd.length-1];
			int td = -1;
			for (int i = 0, length_ptd = ptd.length; i < length_ptd; i++){
				if(Double.compare(ptd[i], u) >= 0){
					td = i+1;
					break;
				}
			}
			
			assert(td>=1);
			TdArray[d] = td;
			
			//randomly initialize
			
			Set<Integer> Zd = new HashSet<Integer>();
			while ( Zd.size() != td ){
				int u_z = rg.nextInt(numTopic);
				Zd.add(u_z);
			}
			ZdMap.put(d, Zd);
			normalCountZd(Zd, +1);
			
			Object[] ZdList = new Object[td];
			ZdList =  Zd.toArray();
			int[] termIDArray = docToWordIDListMap.get(d);
			for (int w = 0, num_word = termIDArray.length; w < num_word; w++){
				int topic_index = rg.nextInt(td);
				int topic = (int) ZdList[topic_index];
				int word = termIDArray[w];
				assignmentListMap.get(d)[w] = topic;
				normalCountWord(topic, word, +1);
			}
		}
		System.out.println("finish init_MU!");
	}

	private static long getCurrTime() {
		return System.currentTimeMillis();
	}
	
	// search space pruning strategy, see the paper
	public int[][] get_top10k(){
		int[][] TopK = new int[numDoc][10];
		Map<Integer, Double> Pdz = null;
		for(int d = 0; d < numDoc; d++){
			Pdz = new HashMap<Integer, Double>();
			for(int k = 0; k < numTopic; k++){
				Pdz.put(k, pdz[d][k]);
			}
			
			ArrayList<Entry<Integer, Double>> l = new ArrayList<Entry<Integer, Double>>(Pdz.entrySet());
			Collections.sort(l, new Comparator<Map.Entry<Integer, Double>>() {
				public int compare(Map.Entry<Integer, Double> o1, Map.Entry<Integer, Double> o2){
					return -Double.compare(o1.getValue(), o2.getValue());
				}
			});
			
			for(int i = 0; i < 10; i++){
				TopK[d][i] = l.get(i).getKey();
			}
		}
		return TopK;
	}
	
	private static int factorial(int n){
		int value = 1;
		while ( n > 0 ){
			value *= n;
			n--;
		}
		
		return value;
	}
	
	private int[][] ZdSearchSize(){
		int count = 0;
		int[] boundary = new int[maxTd];
		for ( int i = 0; i < maxTd; i++ ){
			int temp = 1;
			int factorial = factorial(i+1);
			for ( int j = 0; j < i+1; j++ ){
				temp *= (searchTopK - j);
			}
			
			count += temp/factorial;
			boundary[i] = count;
		}
		
		int[][] array = new int[count][];
		for ( int i = 0; i < count; i++ ){
			for ( int j = 0; j < boundary.length; j++ ){
				if ( i < boundary[j] ){
					array[i] = new int[j+1];
					break;
				}
			}
		}
		
		return array;
	}
	

	// for searching space
	public int[][] getTopKTopics(int[][] docTopKTopics){
		Set<Integer> topKTopics = new HashSet<Integer>();
		int minIndex = -1;
		double minValue = 2;
		for(int d = 0; d < numDoc; d++){
			minValue = 2;
			minIndex = -1;
			topKTopics.clear();
			
			for(int k = 0; k < numTopic; k++){
				if ( topKTopics.size() < searchTopK ){
					topKTopics.add(k);
					if ( Double.compare(minValue, pdz[d][k]) > 0 ){
						minValue = pdz[d][k];
						minIndex = k;
					}
				} else {
					if (Double.compare(minValue, pdz[d][k]) < 0) {
						topKTopics.remove(minIndex);
						topKTopics.add(k);
						minIndex = minPDZTopicIndex(d, topKTopics);
						minValue = pdz[d][minIndex];
					}
				}
			}
			
			int index = 0;
			for ( int topic : topKTopics ){
				docTopKTopics[d][index++] = topic;
			}
		}
		
		return docTopKTopics;
	}
	
	
	private int minPDZTopicIndex(int doc, Set<Integer> topics){
		double min = 2;
		int minIndex = -1;
		for ( int topic : topics ){
			if ( Double.compare(min, pdz[doc][topic]) > 0 ){
				min = pdz[doc][topic];
				minIndex = topic;
			}
		}
		
		return minIndex;
	}

	public void run_iteration(String flag) {
		
		int[][] topicSettingArray = ZdSearchSize();
		int[][] docTopKTopics = new int[numDoc][searchTopK];
		double[] Ptd_Zd = new double[topicSettingArray.length];
		int[] termIDArray = null;
		int[][] mediateSamples = null;
		
		Map<Integer, int[][]> mediateSampleMap = 
			new HashMap<Integer, int[][]>();
		
		for ( int i = 0; i < numDoc; i++ ){
			termIDArray = docToWordIDListMap.get(i);
			mediateSamples = 
				new int[topicSettingArray.length][termIDArray.length];
			mediateSampleMap.put(i, mediateSamples);
		}
		
		for (int iteration = 1; iteration <= numIter; iteration++) {
			System.out.println(iteration + "th iteration begin");
			if((iteration%saveStep)==0){
				saveModel(flag+"_iter"+iteration);
			}
			
			long _s = getCurrTime();
			
			compute_phi();
			compute_pz();
			compute_pzd();
			
			docTopKTopics = getTopKTopics(docTopKTopics);
			
			for (int s = 0; s < numDoc; s++) {
				termIDArray = docToWordIDListMap.get(s);
				int num_word = termIDArray.length;
				Set<Integer> preZd = ZdMap.get(s);
				normalCountZd(preZd, -1);
				mediateSamples = mediateSampleMap.get(s);
				
				for (int w = 0; w < num_word; w++){
					if (use_gpu){
						// use GPU-PDMM
						ratioCount(assignmentListMap.get(s)[w], s, termIDArray[w], w, -1);
						}
					else{
						// use PDMM
						normalCountWord(assignmentListMap.get(s)[w], termIDArray[w], -1);
					}
				}
				
				topicSettingArray = enumerateTopicSetting(
						topicSettingArray, docTopKTopics[s], maxTd);
				int length_topicSettingArray = topicSettingArray.length;
				
				for(int round = 0; round < length_topicSettingArray; round++){
					int[] topicSetting = topicSettingArray[round];
					int length_topicSetting = topicSetting.length;
					
					//Gibbs Sampling
					for (int w = 0; w < num_word; w++){
						int wordID = termIDArray[w];
						double[] pzDist = new double[length_topicSetting];
						for (int index = 0; index < length_topicSetting; index++) {
							int topic = (int) topicSetting[index];
							double pz = 1.0 * (nzw[topic][wordID] + beta) / (nz[topic] + vocSize * beta);
							pzDist[index] = pz;
						}

						for (int i = 1; i < length_topicSetting; i++) {
							pzDist[i] += pzDist[i - 1];
						}

						double u = rg.nextDouble() * pzDist[length_topicSetting - 1];
						int newTopic = -1;
						for (int i = 0; i < length_topicSetting; i++) {
							if (Double.compare(pzDist[i], u) >= 0) {
								newTopic = topicSetting[i];
								break;
							}
						}
						// update
						mediateSamples[round][w] = newTopic;
						if (use_gpu){
							ratioCount(newTopic, s, wordID, w, +1);
						}
						else{
							normalCountWord(newTopic, wordID, +1);
						}
					}
					
					for (int w = 0; w < num_word; w++){
						int wordID = termIDArray[w];
						if (use_gpu){
							ratioCount(mediateSamples[round][w], s, wordID, w, -1);
						}
						else{
							normalCountWord(mediateSamples[round][w], wordID, -1);
						}
					}
				}
				
				for (int round = 0; round < length_topicSettingArray; round++){
					int[] topicSetting = topicSettingArray[round];
					int length_topicSetting = topicSetting.length;
					double p1 = Math.pow(namda, topicSetting.length) * Math.exp(-namda);
					double p21 = 1.0;
					double p22 = 1.0;
					
					for(int k : topicSetting){
						p21*= (alpha + Ck[k]);
					}
					
					for(int i = 0; i < length_topicSetting; i++){
						p22 *= (CkSum + numTopic*alpha - i);
					}
					double p2 = p21/p22;
					double p31 = 1.0;
					double p32 = 1.0;
					Map<Integer, Map<Integer, Integer>> Ndkt = 
						getNdkt_Zd(s, topicSetting, mediateSamples[round]);
					Map<Integer, Integer> Ndk = 
						getNdk_Zd(s, topicSetting, mediateSamples[round]);
					for(int k: topicSetting){
						Set<Integer> dk = 
							getdk_Zd(s, mediateSamples[round], k);
						for(int t: dk){
							for (int i = 0; i < Ndkt.get(k).get(t); i++){
								p31 *= (beta+nzw[k][t]+Ndkt.get(k).get(t)-i);
							}
						}
						for(int j = 0; j < Ndk.get(k); j++){
							p32 *= (nz[k]+vocSize*beta+Ndk.get(k)-j);
						}
						dk.clear();
						
					}
					Ndkt.clear();
					Ndk.clear();
					double p3 = p31/p32;
					Ptd_Zd[round] = p1*p2*p3;
				}
				
				for(int i = 1; i < length_topicSettingArray; i++){
					Ptd_Zd[i]+=Ptd_Zd[i-1];
				}
				
				// sample
				double u_ptdzd = rg.nextDouble()*Ptd_Zd[length_topicSettingArray-1];
				int new_index = -1;
				for (int i = 0; i < length_topicSettingArray; i++) {
					if (Double.compare(Ptd_Zd[i], u_ptdzd) >= 0) {
						new_index = i;
						break;
					}
				}
				
				TdArray[s] = topicSettingArray[new_index].length;
				preZd.clear();
				for(int k: topicSettingArray[new_index]){
					preZd.add(k);
				}
				
				normalCountZd(preZd, +1);
				System.arraycopy(
						mediateSamples[new_index], 0, 
						assignmentListMap.get(s), 0, mediateSamples[new_index].length);
				for(int w = 0; w < termIDArray.length; w++){
					if (use_gpu){
						ratioCount(mediateSamples[new_index][w], s, termIDArray[w], w, +1);
					}
					else{
						normalCountWord(mediateSamples[new_index][w], termIDArray[w], +1);
					}
				}
			}
			long _e = getCurrTime();
			System.out.println(iteration + "th iter finished and every iterration costs " + (_e - _s) + "ms! Snippet "
					+ numTopic + " topics round " + roundIndex);
		}
	}
	
	private static int[][] enumerateTopicSetting(int[][] topicSettingArray,
			int[] topKTopics, int maxTd) {
		// enumerate all the possible topic setting
		// we assume the maxTd is up to 4 (if maxTd is bigger then 4, you should apply your function
		int index = 0;
		if ( maxTd > 0)
			index = enumerateOneTopicSetting(topicSettingArray, topKTopics, index);
		
		if ( maxTd > 1)
		index = enumerateTwoTopicSetting(topicSettingArray, topKTopics, index);
		
		if ( maxTd > 2)
			index = enumerateThreeTopicSetting(topicSettingArray, topKTopics, index);
		
		if ( maxTd > 3)
			index = enumerateFourTopicSetting(topicSettingArray, topKTopics, index);
		
		return topicSettingArray;
	}
	
	private static int enumerateOneTopicSetting(int[][] topicSettingArray,
			int[] topKTopics, int index){
		for ( int i = 0; i < topKTopics.length; i++ ){
			topicSettingArray[index++][0] = topKTopics[i];
		}
		
		return index;
	}
	
	private static int enumerateTwoTopicSetting(int[][] topicSettingArray,
			int[] topKTopics, int index){
		for ( int i = 0; i < topKTopics.length; i++ ){
			for ( int j = i+1; j < topKTopics.length; j++ ){
				topicSettingArray[index][0] = topKTopics[i];
				topicSettingArray[index++][1] = topKTopics[j];
			}
		}
		
		return index;
	}
	
	private static int enumerateThreeTopicSetting(int[][] topicSettingArray,
			int[] topKTopics, int index){
		for ( int i = 0; i < topKTopics.length; i++ ){
			for ( int j = i+1; j < topKTopics.length; j++ ){
				for (int n = j + 1; n < topKTopics.length; n++) {
					topicSettingArray[index][0] = topKTopics[i];
					topicSettingArray[index][1] = topKTopics[j];
					topicSettingArray[index++][2] = topKTopics[n];
				}
			}
		}
		
		return index;
	}
	
	private static int enumerateFourTopicSetting(int[][] topicSettingArray,
			int[] topKTopics, int index){
		for ( int i = 0; i < topKTopics.length; i++ ){
			for ( int j = i+1; j < topKTopics.length; j++ ){
				for (int n = j + 1; n < topKTopics.length; n++) {
					for (int m = n +1; m < topKTopics.length; m++){
					topicSettingArray[index][0] = topKTopics[i];
					topicSettingArray[index][1] = topKTopics[j];
					topicSettingArray[index][2] = topKTopics[n];
					topicSettingArray[index++][3] = topKTopics[m];
					}
				}
			}
		}
		
		return index;
	}

	public void run_GSDMM(String flag) {
		initNewModel();
		init_GSDMM();
		run_iteration(flag);
		saveModel(flag);
	}

	public void saveModel(String flag) {

		compute_phi();
		compute_pz();
		compute_pzd();
		saveModelPz(flag + "_theta.txt");
		saveModelPhi(flag + "_phi.txt");
		saveModelWords(flag + "_words.txt");
		saveModelZd(flag+"_zd.txt");
		saveModelPdz(flag + "_pdz.txt");
		saveAssign(flag + "_assign.txt");
		saveTermAssign(flag + "_termassign.txt");
	}

	private Boolean saveAssign(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename);
			for (int i = 0; i < numDoc; i++) {
				int[] assignments = assignmentListMap.get(i);
				for (int j = 0; j < assignments.length; j++){
					out.print(assignments[j] + " ");
				}
				out.println();
			}
			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving assignments:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}
	
	private Boolean saveTermAssign(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename);
			for (int i = 0; i < numDoc; i++) {
				int[] assignments = assignmentListMap.get(i);
				int[] termIDArray = docToWordIDListMap.get(i);
				for (int j = 0; j < assignments.length; j++){
					out.print(id2word.get(termIDArray[j]) + ":" + assignments[j] + " ");
				}
				out.println();
			}
			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving term assignments:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}


	public void compute_phi() {
		for (int i = 0; i < numTopic; i++) {
			double sum = 0.0;
			for (int j = 0; j < vocSize; j++) {
				sum += nzw[i][j];
			}
			
			for (int j = 0; j < vocSize; j++) {
				phi[i][j] = (nzw[i][j] + beta) / (sum + vocSize * beta);
			}
		}
	}

	public void compute_pz() {
		double sum = 0.0;
		for (int i = 0; i < numTopic; i++) {
			sum += nz[i];
		}
		
		for (int i = 0; i < numTopic; i++) {
			pz[i] = 1.0 * (nz[i] + alpha) / (sum + numTopic * alpha);
		}
	}

	public void compute_pzd() {
		/** calculate P(z|w) **/
		for (int i = 0; i < vocSize; i++) {
			double row_sum = 0.0;
			for (int j = 0; j < numTopic; j++) {
				pwz[i][j] = pz[j] * phi[j][i];
				row_sum += pwz[i][j];
			}
			
			for (int j = 0; j < numTopic; j++) {
				pwz[i][j] = pwz[i][j] / row_sum;
			}
		}

		for (int i = 0; i < numDoc; i++) {
			int[] doc_word_id = docToWordIDListMap.get(i);
			double row_sum = 0.0;
			for (int j = 0; j < numTopic; j++) {
				pdz[i][j] = 0;
				for (int wordID : doc_word_id) {
					pdz[i][j] += pwz[wordID][j];
				}
				row_sum += pdz[i][j];
			}
			
			for (int j = 0; j < numTopic; j++) {
				pdz[i][j] = pdz[i][j] / row_sum;
			}
		}
	}
	
	public Boolean saveModelZd(String filename){
		try {
			PrintWriter out = new PrintWriter(filename);
			int td2 = 0;
			int td3 = 0;
			int td1 = 0;
			int td4 = 0;
			for (int i = 0; i < numDoc; i++) {
				if(ZdMap.get(i).size()==2){
					td2++;
				}
				if(ZdMap.get(i).size()==3){
					td3++;
				}
				if(ZdMap.get(i).size()==1){
					td1++;
				}
				if(ZdMap.get(i).size()==4){
					td4++;
				}
				Iterator it = ZdMap.get(i).iterator();
				while(it.hasNext()){
					out.print(it.next() + " ");
				}
				out.println();
			}
			out.println("td=1:"+td1+" td=2:"+td2+" td=3:"+td3+" td4:"+td4);

			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving p(z|d) distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}
	

	public boolean saveModelPdz(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename);

			for (int i = 0; i < numDoc; i++) {
				for (int j = 0; j < numTopic; j++) {
					out.print(pdz[i][j] + " ");
				}
				out.println();
			}

			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving p(z|d) distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}

	public boolean saveModelPz(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename);

			for (int i = 0; i < numTopic; i++) {
				out.print(pz[i] + " ");
			}
			out.println();

			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving pz distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}

	public boolean saveModelPhi(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename);

			for (int i = 0; i < numTopic; i++) {
				for (int j = 0; j < vocSize; j++) {
					out.print(phi[i][j] + " ");
				}
				out.println();
			}
			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving word-topic distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}

	public boolean saveModelWords(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename, "UTF8");
			for (String word : word2id.keySet()) {
				int id = word2id.get(word);
				out.println(word + "," + id);
			}
			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saveing words list: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}

	public static void main(String[] args) {
		
		ArrayList<Document> doc_list = Document.LoadCorpus("dataset//Snippet//snippet_data.txt");
		//params here, please refer to our paper
		int numIter = 1000, save_step = 200;
		double beta = 0.1;
		String similarityFileName = "dataset//Snippet//snippet_word_similarity.txt";
		double weight = 0.3; // 0.1 for BaiduQA, 0.3 for Snippet
		double threshold = 0.5; // 0.7 for BaiduQA, 0.5 for Snippet
		int filterSize = 20; //notice we use filtersize=20 for Snippet but 40 for BaiduQA, it's a mistake we make in paper writing!
		boolean use_gpu = true; //use GPU-PDMM or PDMM
		double namda = 1.5;
		int maxTd = 2;
		int Topk = 10;
		
		// you can change the loop to fit your own experiment need
		for (int round = 1; round <= 5;round += 1) {
	//	for (int maxTd = 1; maxTd <= 4; maxTd +=1){
			for (int num_topic = 40; num_topic <= 80; num_topic += 20) {
				//int num_topic = 40;
				double alpha = 1.0 * 50 / num_topic;
				PDMM gsdmm = 
					new PDMM(doc_list, num_topic, numIter, save_step, beta, alpha, namda, threshold);
				gsdmm.use_gpu = use_gpu;
				gsdmm.word2idFileName = "dataset//Snippet//snippet_word2id.txt";
				gsdmm.topWords = 100;
				gsdmm.maxTd = maxTd;
				gsdmm.searchTopK = Topk; // search size for heuristic search , 
				                          //we don't use heuristic search if we set searchTopK = numTopic
				
				gsdmm.filterSize = filterSize;
				gsdmm.roundIndex = round;
				gsdmm.similarityFileName = similarityFileName;
				gsdmm.weight = weight;
				gsdmm.initNewModel();
				gsdmm.init_GSDMM();
				String flag = round+"round_"+num_topic + "topic_qa_";
				flag = "result//"+flag;
				
				gsdmm.run_iteration(flag); 
				try {
					Thread.sleep(10000);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
			}
		}
		
	}
}


/**
 * Comparator to rank the words according to their probabilities.
 */
class TopicalWordComparator implements Comparator<Integer> {
	private double[] distribution = null;

	public TopicalWordComparator(double[] distribution) {
		this.distribution = distribution;
	}

	@Override
	public int compare(Integer w1, Integer w2) {
		return Double.compare(distribution[w1], distribution[w2]);
	}
}
