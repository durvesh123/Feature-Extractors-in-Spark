import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;


public class CrossValidation {

    private SparkConf conf;
    private SparkContext sc;
    private String inputFile;
    private SparkSession spark;
    private Dataset<Row> df;

    public CrossValidation(){
        conf = new SparkConf().setAppName("CV").setMaster("local");
        sc = new SparkContext(conf);
        inputFile = "/Users/durveshvedak/IdeaProjects/FeatureExtractorsSpark/src/main/resources/titanic-train.csv";
        spark = SparkSession.builder().appName("CV").getOrCreate();
    }

    public static void main(String args[]) {
        CrossValidation cv = new CrossValidation();
        cv.df = cv.cleanData(cv.loadDataset(cv.inputFile));
        Dataset<Row> cleanAndEncodedData = cv.encodeData(cv.df);
        Dataset<Row> train_df = cv.prepareForTraining(cleanAndEncodedData);
        Dataset<Row> readForTraining = train_df.withColumnRenamed("Survived", "label");
        cv.kFoldCV(3,readForTraining);
        //cv.kFoldCV(5,readForTraining);
        //cv.kFoldCV(10,readForTraining);
        //cv.LOOCV(readForTraining);
    }

    public Dataset<Row> loadDataset(String ip){
        StructType schema = new StructType()
                .add("PassengerId", "double")
                .add("Survived", "double")
                .add("Pclass", "double")
                .add("Name", "string")
                .add("Sex", "string")
                .add("Age", "double")
                .add("SibSp", "double")
                .add("Parch", "double")
                .add("Ticket", "string")
                .add("Fare", "double")
                .add("Cabin", "double")
                .add("Embarked", "string");

        return spark.read().option("mode", "DROPMALFORMED").schema(schema).csv(ip);
    }

    public Dataset<Row> cleanData(Dataset<Row> df){
        Dataset<Row> df_1 = df.na().fill(0, new String[]{"Cabin", "Age","PassengerId","Survived","Pclass","SibSp","Parch","Fare"});
        Dataset<Row> df_2 = df_1.na().fill("0", new String[]{"Name","Sex","Ticket","Embarked"});
        return  df_2;
    }

    public Dataset<Row> encodeData(Dataset<Row> df){

        //name encoder
        StringIndexerModel name = new StringIndexer()
                .setInputCol("Name")
                .setOutputCol("Name_temp")
                .fit(df);
        Dataset<Row> indexed1 = name.transform(df);

        OneHotEncoder encoder1 = new OneHotEncoder()
                .setInputCol("Name_temp")
                .setOutputCol("Name_Vec");

        Dataset<Row> encoded1 = encoder1.transform(indexed1);


        //sex encoder
        StringIndexerModel sex = new StringIndexer()
                .setInputCol("Sex")
                .setOutputCol("Sex_temp")
                .fit(df);
        Dataset<Row> indexed2 = sex.transform(encoded1);

        OneHotEncoder encoder2 = new OneHotEncoder()
                .setInputCol("Sex_temp")
                .setOutputCol("Sex_Vec");

        Dataset<Row> encoded2 = encoder2.transform(indexed2);

        //ticket encoder
        StringIndexerModel ticket = new StringIndexer()
                .setInputCol("Ticket")
                .setOutputCol("Ticket_temp")
                .fit(df);
        Dataset<Row> indexed3 = ticket.transform(encoded2);

        OneHotEncoder encoder3 = new OneHotEncoder()
                .setInputCol("Ticket_temp")
                .setOutputCol("Ticket_Vec");

        Dataset<Row> encoded3 = encoder3.transform(indexed3);

        //Embarked encoder
        StringIndexerModel embarked = new StringIndexer()
                .setInputCol("Embarked")
                .setOutputCol("Embarked_temp")
                .fit(df);
        Dataset<Row> indexed4 = embarked.transform(encoded3);

        OneHotEncoder encoder4 = new OneHotEncoder()
                .setInputCol("Embarked_temp")
                .setOutputCol("Embarked_Vec");

        Dataset<Row> encoded4 = encoder4.transform(indexed4);

       return encoded4.select("Survived","PassengerId","Pclass","Age","SibSp","Parch","Fare","Cabin","Name_Vec","Sex_Vec","Ticket_Vec","Embarked_Vec");
    }

    public Dataset<Row> prepareForTraining(Dataset<Row> df){
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"PassengerId","Pclass","Name_Vec","Sex_Vec","Age","SibSp","Parch","Ticket_Vec","Fare","Cabin","Embarked_Vec"})
                .setOutputCol("features");
        Dataset<Row> output = assembler.transform(df);
        return output.select("features","Survived");
    }

    public void kFoldCV(int k,Dataset<Row>df){
        Dataset<Row>[]train;
        switch(k) {
            case 3:
                train = df.randomSplit(new double[]{0.3, 0.3, 0.3});
                        trainPredictMetric(train[0],train[1].union(train[2]));
                break;
            case 5:
                train = df.randomSplit(new double[]{0.2, 0.2, 0.2, 0.2, 0.2});
                        trainPredictMetric(train[0],train[1].union(train[2].union(train[3].union(train[4]))));
                break;
            case 10:
                train = df.randomSplit(new double[]{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1});
                        trainPredictMetric(train[0],train[1].union(train[2].union(train[3].union(train[4].union(train[5].union(train[6].union(train[7].union(train[8].union(train[9])))))))));
                break;
        }

    }

    public void LOOCV(Dataset<Row>df){
        Iterator<Row> itr = df.toLocalIterator();
        while(itr.hasNext()){
            Row r = itr.next();
            Dataset<Row>train = df.filter(row->!row.equals(r));
            Dataset<Row>test = df.filter(row->row.equals(r));
            trainPredictMetric(train,test);
        }
    }

    public void trainPredictMetric(Dataset<Row>train,Dataset<Row>test){

        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);

        LogisticRegressionModel lrModel = lr.fit(train);

        Iterator<Row> itr = test.toLocalIterator();
        List<Vector> probabilties = new ArrayList<>();
        while(itr.hasNext()){
            Row r = itr.next();
            Vector v= (Vector) r.get(0);
            probabilties.add(lrModel.predictProbability(v));
        }
        System.out.println("Probabilities"+probabilties.toString());


        LogisticRegressionTrainingSummary trainingSummary = lrModel.summary();
        double[] objectiveHistory = trainingSummary.objectiveHistory();
        for (double lossPerIteration : objectiveHistory) {
            System.out.println(lossPerIteration);
        }

        BinaryLogisticRegressionSummary binarySummary = (BinaryLogisticRegressionSummary) trainingSummary;

        Dataset<Row> roc = binarySummary.roc();
        roc.show();
        roc.select("FPR").show();
        System.out.println(binarySummary.areaUnderROC());


    }
}

