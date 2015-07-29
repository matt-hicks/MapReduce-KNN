# MapReduce-KNN
K nearest neighbour implementation for Hadoop MapReduce

This is a java program designed to work with the MapReduce framework.
In this example the K nearest neighbour classification method (supervised machine learning)
is applied to some sample data about car types and buyer characteristics, so that it classifies
a buyer with a likely car model.

Usage:
hadoop jar KnnPattern.jar KnnPattern /home/mhi/knn/CarOwners.csv /home/mhi/knn/res /home/mhi/knn/KnnParams.txt
Explanation of this command:
KnnPattern.jar – the jar file containing the source code.
KnnPattern – the top level class in the program, containing the Mapper and Reducer classes and the main() method.
1st argument: /home/mhi/knn/CarOwners.csv – the location in HDFS of the data input file.
2nd argument: /home/mhi/knn/res – the output directory in HDFS.
3rd argument: /home/mhi/knn/KnnParams.txt - the location in HDFS of a parameter file with the following format:
K, Age, Income, Status, Gender, Children
E.g. 5, 67, 16668, Single, Male, 3
Where K is the number of nearest neighbours to consider in the classification.
