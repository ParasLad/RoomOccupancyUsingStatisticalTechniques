#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd


# capturing data from the txt file of dataset
def Dataextract(file):
    my_file = open(file)
    content = my_file.readlines()
    content = [x.strip() for x in content]

    l1 = []

    for i in content:
        l = i.split(",")
        l1.append(l)

    #formulating each input and output in unique list from dataset
    leng1 = len(l1)
    leng2 = len(l1[1])
    temp = []
    hum = []
    light = []
    co2 = []
    humratio = []
    occ = []

    for i in range(1,leng1):
        for j in range(2,leng2):
            if j == 2:
                temp.append(float(l1[i][j]))
            if j==3:
                hum.append(float(l1[i][j]))
            if j==4:
                light.append(float(l1[i][j]))
            if j==5:
                co2.append(float(l1[i][j]))
            if j==6:
                humratio.append(float(l1[i][j]))
            if j==7:
                occ.append(float(l1[i][j]))

    # normalization od data to range 0 to 1            
    l_temp = min(temp)
    m_temp = max(temp)

    l_hum = min(hum)
    m_hum = max(hum)

    l_light = min(light)
    m_light = max(light)

    l_co2 = min(co2)
    m_co2 = max(co2)

    l_humratio = min(humratio)
    m_humratio = max(humratio)

    new_max = 1
    new_min = 0

    for i in range(len(temp)):
        temp[i] = (((temp[i]-l_temp)/(m_temp-l_temp))*(new_max-new_min))+new_min
        hum[i] = (((hum[i]-l_hum)/(m_hum-l_hum))*(new_max-new_min))+new_min
        light[i] = (((light[i]-l_light)/(m_light-l_light))*(new_max-new_min))+new_min
        co2[i] = (((co2[i]-l_co2)/(m_co2-l_co2))*(new_max-new_min))+new_min
        humratio[i] = (((humratio[i]-l_humratio)/(m_humratio-l_humratio))*(new_max-new_min))+new_min
    data_trn = []
    for i in range(len(temp)):
        u = [temp[i],hum[i],light[i],co2[i],humratio[i],occ[i]]
        data_trn.append(u)

    data_trn = np.array(data_trn)
    data = { "Temperature" : temp, "Humidity" : hum, "Light" : light, "CO2" : co2, "Humidity Ratio" : humratio}
    return(data_trn,data)



#Fetching Training data 
data_trn,data = Dataextract("C:/Users/Paras/Downloads/occupancy_data/datatraining.txt")
data_trnsamp = []
data_trnlab = []
for i in data_trn:
    data_trnsamp.append(i[:-1])
    #data_trnsamp.append([i[0],i[1],i[3]])
    data_trnlab.append([i[-1]])
data_trnsamp =  np.array(data_trnsamp)
data_trnlab = np.array(data_trnlab)



#Fetching Testing data when door is closed 
data_test,data1 = Dataextract("C:/Users/Paras/Downloads/occupancy_data/datatest.txt")
data_testsamp = []
data_testlab = []
for i in data_test:
    data_testsamp.append(i[:-1])
    #data_testsamp.append([i[0],i[1],i[3]])
    data_testlab.append([i[-1]])
data_testsamp =  np.array(data_testsamp)
data_testlab = np.array(data_testlab)


#Fetching Testing data when door is open
data_test2,data2 = Dataextract("C:/Users/Paras/Downloads/occupancy_data/datatest2.txt")
data_testsamp2 = []
data_testlab2 = []
for i in data_test2:
    data_testsamp2.append(i[:-1])
    #data_testsamp2.append([i[0],i[1],i[3]])
    data_testlab2.append([i[-1]])
data_testsamp2 =  np.array(data_testsamp2)
data_testlab2 = np.array(data_testlab2)



#plotting correlation matrix
df = pd.DataFrame(data,columns=['Temperature','Humidity','Light','CO2','Humidity Ratio'])
print(df.corr())


#Logistic Regression Class
class LogisticRegression():
    def fit(self,X,Y,B):
        a = np.dot(X,B)
        P = 1 / (1 + np.exp(-a))
        W = np.zeros((P.shape[0],P.shape[0]))
        for i in range(len(P)):
            for j in range(len(P)):
                if i == j:
                    W[i][j] = P[i]*(1-P[i])
        H = X.T.dot(W).dot(X);
        G = np.dot(X.T, (Y-P));
        up = np.dot(np.linalg.pinv(H), G)
        B = B + up
        return B;
    def predict(self,X,B):
        Y = []
        for i in range(len(X)):
            a = np.dot(X[i].T,B)
            out = 1/ (1+np.exp(-a))
            if(out<0.5):
                Y.append(0)
            else:
                Y.append(1)
        return Y


#Decision Tree Classifier
class N():
    def __init__(self, feat_ind=None, thr=None, left=None, right=None, information_gain=None, value=None):        
        self.feat_ind = feat_ind
        self.thr = thr
        self.left = left
        self.right = right
        self.information_gain = information_gain
        self.value = value
		
		
class DecisionTreeClassifier():
    def __init__(self, min_samp_split=2, maximumdepth=2): 
        self.r = None
        self.min_samp_split = min_samp_split
        self.maximumdepth = maximumdepth
        
    def make_tree(self, data, current_depth=0):
        X, Y = data[:,:-1], data[:,-1]
        samples, features = np.shape(X)
        
        # split until we met stopping condition
        if samples>=self.min_samp_split and current_depth<=self.maximumdepth:
            # fetching the best split
            best_split = self.fetch_best_split(data, samples, features)
            # verifying if information gain is positive
            if best_split["information_gain"]>0:
                # loop left
                l_subtree = self.make_tree(best_split["left_data"], current_depth+1)
                # loop right
                r_subtree = self.make_tree(best_split["right_data"], current_depth+1)
                # return decision node
                return N(best_split["feat_ind"], best_split["thr"], 
                            l_subtree, r_subtree, best_split["information_gain"])
        
        # calculate leaf node
        leaf_value = self.retrieve_leaf_val(Y)
        # return leaf node
        return N(value=leaf_value)
    
    def fetch_best_split(self, data, samples, features):
        best_split = {}
        maximum_IG = -float("inf")
        
        # go through all the features
        for feat_ind in range(features):
            feature_values = data[:, feat_ind]
            poss_th = np.unique(feature_values)
            # loop over all the feature values present in the data
            for thr in poss_th:
                # fetching current split
                left_data, right_data = self.split(data, feat_ind, thr)
                if len(left_data)>0 and len(right_data)>0:
                    y, left_y, right_y = data[:, -1], left_data[:, -1], right_data[:, -1]
                    # calculate information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>maximum_IG:
                        best_split["feat_ind"] = feat_ind
                        best_split["thr"] = thr
                        best_split["left_data"] = left_data
                        best_split["right_data"] = right_data
                        best_split["information_gain"] = curr_info_gain
                        maximum_IG = curr_info_gain
                        
        return best_split
    
    def split(self, data, feat_ind, thr):
        ''' function to split the data '''
        
        left_data = np.array([row for row in data if row[feat_ind]<=thr])
        right_data = np.array([row for row in data if row[feat_ind]>thr])
        return left_data, right_data
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def retrieve_leaf_val(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        data = np.concatenate((X, Y), axis=1)
        self.r = self.make_tree(data)
    
    def predict(self, X):
        ''' function to predict new data '''
        
        preditions = [self.make_prediction(x, self.r) for x in X]
        return preditions
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feat_ind]
        if feature_val<=tree.thr:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)



#SVM Classifier
class SVM:
    def __init__(self, LR=0.001, Lambda=0.001, Iter=500):
        self.lr = LR
        self.Lambda = Lambda
        self.iters = Iter
        self.W = None
        self.B = None


    def fit(self, X, Y):
        data, features = X.shape
        
        up_y = np.where(Y <= 0, -1, 1)
        
        self.W = np.zeros(features)
        self.B = 0

        for _ in range(self.iters):
            for ID, Xi in enumerate(X):
                cond = up_y[ID] * (np.dot(Xi, self.W) - self.B) >= 1
                if cond:
                    self.W -= self.lr * (2 * self.Lambda * self.W)
                else:
                    self.W -= self.lr * (2 * self.Lambda * self.W - np.multiply(Xi, up_y[ID]))
                    self.B -= self.lr * up_y[ID]


    def predict(self, X):
        val = np.dot(X, self.W) - self.B
        out = []
        for i in val:
            if i<=0:
                out.append(0)
            else:
                out.append(1)
        return out


#calculating accuracy
def accuracy(pred, data):
    mis_classified = 0
    for i in range(len(data)):
        if (data[i][0] != pred[i]):
            mis_classified +=1
    acc = 1 - mis_classified/len(data)
    return(acc)



#Performing Logistic Regression over our dataset

b = np.zeros(data_trnsamp.shape[1])
B = []
for i in b:
    B.append([i])
B = np.array(B)
lr = LogisticRegression()
for i in range(20):
    B = lr.fit(data_trnsamp,data_trnlab,B)



out_pred= lr.predict(data_trnsamp,B)
out_trn = accuracy(out_pred,data_trnlab)
out_pred= lr.predict(data_testsamp,B)
out_test1 = accuracy(out_pred,data_testlab)
out_pred= lr.predict(data_testsamp2,B)
out_test2 = accuracy(out_pred,data_testlab2)
print("Logistic Regression")
print("Training Accuracy : ",out_trn)
print("Testing Accuracy when door is closed: ",out_test1)
print("Testing accuracy when door is open: ",out_test2)


#Performing Decision Tree over our dataset

classifier = DecisionTreeClassifier(maximumdepth=3)
classifier.fit(data_trnsamp,data_trnlab)

Y_pred = classifier.predict(data_trnsamp) 
out_trn = accuracy(Y_pred,data_trnlab)
Y_pred = classifier.predict(data_testsamp) 
out_test1 = accuracy(Y_pred,data_testlab)
Y_pred = classifier.predict(data_testsamp2) 
out_test2 = accuracy(Y_pred,data_testlab2)
print("Decision Tree Classifier")
print("Training Accuracy : ",out_trn)
print("Testing Accuracy when door is closed: ",out_test1)
print("Testing accuracy when door is open: ",out_test2)


#Performing Support Vector Machine our our dataset

svm = SVM()
svm.fit(data_trnsamp,data_trnlab)

y_pred = svm.predict(data_trnsamp)
out_trn = accuracy(y_pred,data_trnlab)
y_pred = svm.predict(data_testsamp)
out_test1 = accuracy(y_pred,data_testlab)
y_pred = svm.predict(data_testsamp2)
out_test2 = accuracy(y_pred,data_testlab2)
print("Support Vector Machine")
print("Training Accuracy : ",out_trn)
print("Testing Accuracy when door is closed: ",out_test1)
print("Testing accuracy when door is open: ",out_test2)

