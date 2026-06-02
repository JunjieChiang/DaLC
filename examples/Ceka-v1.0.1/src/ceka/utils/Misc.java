package ceka.utils;

import java.io.File;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class Misc {
	
	/**
	 * whether two double values are same
	 * @param d1 value1
	 * @param d2 value 2
	 * @param e  the minimum difference if two values are not same
	 * @return true or false
	 */
	public static boolean isDoubleSame(double d1, double d2, double e) {
		double diff = d1 - d2;
		if (Math.abs(diff) <= e)
			return true;
		return false;
	}
	
	/**
	 * round
	 * @param d
	 * @param decimalPlace
	 * @return
	 */
	public static Double round(double d, int decimalPlace) {
		// see the Javadoc about why we use a String in the constructor
		// http://java.sun.com/j2se/1.5.0/docs/api/java/math/BigDecimal.html#BigDecimal(double)
		BigDecimal bd = new BigDecimal(Double.toString(d));
		bd = bd.setScale(decimalPlace, BigDecimal.ROUND_HALF_UP);
		return bd.doubleValue();
	}
	
	/**
	 * extract file name with/without suffix
	 * @param path
	 * @param suffix whether need suffix
	 * @return file name
	 */
	public static String exstractFileName(String path, boolean suffix) {
		int begin = path.lastIndexOf('/');
		if (begin == -1)
			begin = path.lastIndexOf('\\');
		if (begin == -1)
			begin = 0;
		else
			begin += 1;
		int lastdot = path.lastIndexOf('.');
		if ((lastdot == -1) || (suffix))
			lastdot = path.length();
		return path.substring(begin, lastdot);
	}
	
	/**
	 * extract extension name of a path
	 * @param path
	 * @return extension name
	 */
	public static String extractFileSuffix(String path) {
		String filename = exstractFileName(path, true);
		int lastdot = filename.lastIndexOf('.');
		return filename.substring(lastdot + 1, filename.length());
	}
	
	/**
	 * extract directory from a path
	 * @param path
	 * @return the directory
	 */
	public static String extractDir(String path) {
		int lastSlash = path.lastIndexOf('/');
		if (lastSlash == -1)
			lastSlash = path.lastIndexOf('\\');
		if (lastSlash == -1)
			return path;
		return path.substring(0, lastSlash + 1);
	}
	
	/**
	 * get an object from a List by provided id
	 * @param list
	 * @param key
	 * @return object or null if not found
	 */
	public static <E extends IdDecorated, K> E getElementById(List<E> list, K key) {
		E retValue = null;
		for (E e: list) {
			if (e.getId().equals(key)) {
				retValue = e;
				break;
			}
		}
		return retValue;
	}
	
	/**
	 *  remove an object in a List 
	 * @param list
	 * @param elem
	 */
	public static <E extends IdDecorated, K> void delElementById(List<E> list, K key) {
		E obj = getElementById(list, key);
		if (obj != null)
			list.remove(obj);
	}
	
	/**
	 * add an object to a List if not existed checking by provided id
	 * @param list
	 * @param elem
	 * @return 
	 */
	public static <E extends IdDecorated> void addElementById(List<E> list, E elem) {
		E obj = getElementById(list, elem.getId());
		if (obj != null)
			list.add(elem);
	}
	
	/**
	 * get an object from a List by equals function
	 * @param list
	 * @param elem
	 * @return object or null if not found
	 */
	public static <E extends Object> E getElementEquals(List<E> list, E elem) {
		E retValue = null;
		for (E e: list) {
			if (e.equals(elem)) {
				retValue = e;
				break;
			}
		}
		return retValue;
	}
	
	/**
	 * ad an object to a List if not existed checking by equals function
	 * @param list
	 * @param elem
	 * @return
	 */
	public static <E extends Object> void addElementIfNotExistedEquals(List<E> list, E elem) {
		E obj = getElementEquals(list, elem);
		if (obj == null)
			list.add(elem);
	}
	
	/**
	 * randomly split a list into two lists based on the element number of first list.
	 * @param originalList
	 * @param numFirstList the element number of first list
	 * @return
	 */
	public static <E extends Object> ArrayList<List<E>> splitRandom(List<E> originalList, int numFirstList) {
		ArrayList<List<E>> lists = new ArrayList<List<E>>();
		List<E> first = null;
		List<E> second = null;
		if (originalList instanceof LinkedList) {
			first = new LinkedList<E>();
			second = new LinkedList<E>();
		} else {
			first = new ArrayList<E>();
			second = new ArrayList<E>();
		}
		lists.add(first);
		lists.add(second);
		ArrayList<Integer> firstIndices = new ArrayList<Integer>();
		long seed = System.currentTimeMillis() + (long)(Math.random() * MAGIC);
		Random rand = new Random(seed);
		while (firstIndices.size() < numFirstList) {
			Integer nextInt = new Integer(rand.nextInt(originalList.size()));
			boolean inFirst = false;
			for (Integer elem: firstIndices) {
				if (elem.intValue() == nextInt.intValue()) {
					inFirst = true;
					break;
				}
			}
			if (!inFirst)
				firstIndices.add(nextInt);
		}
		int index = 0;
		for (E e : originalList) {
			boolean inFirst = false;
			for (Integer elem: firstIndices) {
				if (elem.intValue() == index) {
					inFirst = true;
					break;
				}
			}
			if (inFirst)
				first.add(e);
			else
				second.add(e);
			index++;
		}
		return lists;
	}
	
	/**
	 * randomly split a list into two lists with the cut point.
	 * @param originalList
	 * @param cut the proportion of the first list
	 * @return
	 */
	public static <E extends Object> ArrayList<List<E>> splitRandom(List<E> originalList, double cut) {
		int numFirst = (int) (originalList.size() * cut);
		return splitRandom(originalList, numFirst);
	}
	
	/**
	 * create a directory
	 * @param dirPath
	 */
	public static void createDirectory(String dirPath) {
		File tempDatasetDir  = new File(dirPath);
		if (!tempDatasetDir.exists())
			tempDatasetDir.mkdirs();
	}
	
	/**
	 * randomly select {num} number of non-repeated integers between interval [min, max]  
	 * @param num number of selected integers
	 * @param min minimum
	 * @param max maximum
	 * @return a list of the selected integers
	 */
	public static ArrayList<Integer> randSelect(int num, int min, int max) {
		int diff = max - min;
		Random rand = new Random();
		ArrayList<Integer> list = new ArrayList<Integer> ();
		
		int count = 0;
		while (count < num) {
			Integer next = new Integer(rand.nextInt(diff + 1));
			if (getElementEquals(list, next) == null) {
				list.add(next);
				count++;
			}
		}
		
		for (int i = 0; i < list.size(); i++) {
			Integer rst = new Integer(list.get(i) + min);
			list.set(i, rst);
		}
		
		return list;
	}
	
	private static final int MAGIC = 0xCE66D;
}
