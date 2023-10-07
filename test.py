import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from collections import Counter
from prettytable import PrettyTable
import sys

class KNN:
    def __init__(self,encoder_type,K,distance_metric):
        self.encoder_type = encoder_type
        self.K = K
        self.distance_metric = distance_metric
    def change_k(self,val):
        self.K=val
    def change_encoder(self,enc):
        self.encoder_type=enc
    def change_distance_metric(self,dm):
        self.distance_metric=dm
    def get_k(self):
        return self.K
    def get_encoder(self):
        return self.encoder_type
    def get_distance_metric(self):
        return self.distance_metric
    def opt_eval(self,data,test_dt):
        np.random.shuffle(data)
        train_data = data
        test_data = test_dt
        x_train=None
        x_test=None
        if self.encoder_type=="VIT":
            x_train=train_data[:,2]
            x_test=test_data[:,2]
        elif self.encoder_type=="ResNet":
            x_train=train_data[:,1]
            x_test=test_data[:,1]
        y_train=train_data[:,3]
        y_test=test_data[:,3]
        x_train = np.array([np.array(x) for x in x_train])
        x_test = np.array([np.array(x) for x in x_test])
        x_train = x_train.reshape(x_train.shape[0],-1)
        x_test = x_test.reshape(x_test.shape[0],-1)
        distance=None
        if self.distance_metric=="Euclidean":
            a=(x_test**2).sum(axis=1)[:,None]
            b=(x_train**2).sum(axis=1)
            c=x_test.dot(x_train.T)
            distance=np.sqrt(abs(a+b-2*c))
        elif self.distance_metric=="Manhattan":
            distance=(np.abs((x_train[:,None]-x_test)).sum(-1)).T
        elif self.distance_metric=="Cosine":
            x_test_norm = np.sqrt((x_test**2).sum(axis=1))[:, np.newaxis]
            x_train_norm = np.sqrt((x_train**2).sum(axis=1))[np.newaxis, :]
            cosine_sim = x_test.dot(x_train.T) / (x_test_norm * x_train_norm)
            distance = 1 - cosine_sim
        prediction=[]
        for sample in distance:
            nearest_indices=np.argsort(sample)[:self.K]
            nearest_labels = y_train[nearest_indices]
            label_counts = Counter(nearest_labels)
            most_common_label = label_counts.most_common(1)[0][0]
            prediction.append(most_common_label)
        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction,average='macro',zero_division=1)
        recall = recall_score(y_test, prediction,average='macro',zero_division=1)
        f1_macro = f1_score(y_test, prediction,average='macro')
        f1_micro=f1_score(y_test, prediction,average='micro')
        return accuracy,precision,recall,f1_macro,f1_micro
    
test_data_file=sys.argv[1]
train_data=np.load('data.npy',allow_pickle=True)
# train_data=train_data[:500,:]
test_data=np.load(test_data_file,allow_pickle=True)
# test_data=test_data[500:600,:]
knn_classifier=KNN("VIT",9,"Euclidean")

encoders=["ResNet","VIT"]
k_vals=range(1,40)
distance_metrics=["Euclidean","Manhattan","Cosine"]
acc=[]
for e in encoders:
    for k in k_vals:
        for d in distance_metrics:
            knn_classifier.change_k(k)
            knn_classifier.change_encoder(e)
            knn_classifier.change_distance_metric(d)
            x=knn_classifier.opt_eval(train_data,test_data)
            acc.append([x,e,k,d])
sorted_list=sorted(acc, key=lambda x: x[0][0], reverse=True)
for i in range(20):
    print("Rank - "+str(i+1))
    print()
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.add_row(["Accuracy",sorted_list[i][0][0]])
    table.add_row(["Precision",sorted_list[i][0][1]])
    table.add_row(["Recall",sorted_list[i][0][2]])
    table.add_row(["F1 Score (Macro)",sorted_list[i][0][3]])
    table.add_row(["F1 Score (Micro)",sorted_list[i][0][4]])
    table.add_row(["Encoder",sorted_list[i][1]])
    table.add_row(["K",sorted_list[i][2]])
    table.add_row(["Distance Metric",sorted_list[i][3]])
    print(table)
    print("---------------------")
    print()