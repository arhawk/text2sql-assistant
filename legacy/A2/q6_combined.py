import json
# This is the function you need to implement
def read_data(filename):
    """Read the data from a json file.

    Keyword arguments:
    filename -- the name of a json file
    """
    # TODO
    i=0

    str_lst=[]
    temp_str=""
    keys={"train":[],"dev":[],"test":[]}
    sql=set()
    for line in open(filename):
        line=line.strip()
        if not line=='[' or line==']':
            if line=="}," or line=="}":
                temp_str+="}"
                str_lst.append(temp_str)
                temp_str=""
            else:
                temp_str+=line
    for ele in str_lst:
        data = json.loads(ele)
        keys[data["data"]].append((data["question"],data["sql"]))
        sql.add(data["sql"])
    return keys,sql

# This is the class you need to implement
class CodeModel:
    def __init__(self, labels, training_data):
        """Prepare the class member variables.
        Save the labels in self.labels and initialise all the weights to 0.

        Keyword arguments:
        labels -- a set of strings, each string is one SQL query
        training_data -- a list, each item is a tuple containing a question and an SQL query
        """
        # TODO
        self.labels=labels
        self.weights={}
        for quest, sql in training_data:
            for word in quest.split(): 
                self.weights[(word, sql)] = 0
        return 
    
    def get_features(self, question, label):
        """Produce a list of features for a specific question and label.
        
        Keyword arguments:
        question -- a string, an English question
        label -- a string, an SQL query
        """
        # TODO
        return [(word,label) for word in question.split()]

    def get_score(self, question, label):
        """Calculate the model's score for a question, label pair.
        
        Keyword arguments:
        question -- a string, an English question
        label -- a string, an SQL query
        """
        # TODO
        score=0
        features=self.get_features(question, label)
        for feature in features:
            score+=self.weights.get(feature,0)
        return score

    def update(self, question, label, change):
        """Modify the model.
        Changes all weights for features for the (question, SQL query) pair by the amount indicated.

        Keyword arguments:
        question -- a string, an English question
        label -- a string, an SQL query
        change -- an integer, how much to change the weights
        """
        # TODO
        features=self.get_features(question, label)
        for feature in features:
            self.weights[feature]=self.weights.get(feature,0)+change
        return

# This is the function you need to implement
def find_best_code(question, model):
    """Predicts the SQL for a question by using a model to try all possible labels.

    Keyword arguments:
    question -- a string, the English question
    model -- a CodeModel, as defined in the Model question
    """
    # TODO
    best_label=next(iter(model.labels))
    max_score=model.get_score(question,best_label)
    for label in model.labels:
        score=model.get_score(question,label)
        if score>max_score:
            max_score=score
            best_label=label
    return best_label

# This is the function you need to implement
def learn(question, answer, model, find_best_code):
    """Updates a model by predicting the SQL for a question and making a Perceptron update 

    Keyword arguments:
    question -- a string, the English question
    answer -- a string, the correct SQL query for this question 
    model -- a CodeModel, as defined in the Model question
    find_best_code -- a function, the one defined the Inference question
    """
    current_query=find_best_code(question,model)#sql/label
    while current_query!=answer:
        model.update(question, current_query, -1)
        model.update(question, answer, 1)
        current_query=find_best_code(question,model)
    return

# This is the function you need to implement
def get_confusion_matrix(eval_data, model, find_best_code):
    """Creates a confusion matrix by predicting the SQL for a question and recording how the answer compares with the true answer 

    Keyword arguments:
    eval_data -- a list of tuples containing the English question and the true SQL query
    model -- a CodeModel, as defined in the Model question
    find_best_code -- a function, the one defined the Inference question
    """
    # TODO
    confusion_matrix = {}
    #初始化所有可能的 (真实 SQL, 预测 SQL) 组合
    for label1 in model.labels:
        for label2 in model.labels:
            confusion_matrix[(label1,label2)]=0
    for question, true_sql in eval_data:
        predicted_sql=find_best_code(question,model)
        confusion_matrix[(true_sql, predicted_sql)]+=1
    return confusion_matrix

class CodeModel:
    def __init__(self, labels, training_data):
        """Prepare the class member variables.
        Save the labels in self.labels and initialise all the weights to 0.

        Keyword arguments:
        labels -- a set of strings, each string is one SQL query
        training_data -- a list, each item is a tuple containing a question and an SQL query
        """
        # TODO
        self.labels=labels
        self.weights={}
        for quest, sql in training_data:
            for word in quest.split(): 
                self.weights[(word, sql)] = 0
        return 
    
    def get_features(self, question, label):
        """Produce a list of features for a specific question and label.
        
        Keyword arguments:
        question -- a string, an English question
        label -- a string, an SQL query
        """
        # TODO
        return [(word,label) for word in question.split()]

    def get_score(self, question, label):
        """Calculate the model's score for a question, label pair.
        
        Keyword arguments:
        question -- a string, an English question
        label -- a string, an SQL query
        """
        # TODO
        score=0
        features=self.get_features(question, label)
        for feature in features:
            score+=self.weights.get(feature,0)
        return score

    def update(self, question, label, change):
        """Modify the model.
        Changes all weights for features for the (question, SQL query) pair by the amount indicated.

        Keyword arguments:
        question -- a string, an English question
        label -- a string, an SQL query
        change -- an integer, how much to change the weights
        """
        # TODO
        features=self.get_features(question, label)
        for feature in features:
            self.weights[feature]=self.weights.get(feature,0)+change
        return

def find_best_code(question, model):
    """Predicts the SQL for a question by using a model to try all possible labels.

    Keyword arguments:
    question -- a string, the English question
    model -- a CodeModel, as defined in the Model question
    """
    # TODO
    best_label=next(iter(model.labels))
    max_score=model.get_score(question,best_label)
    for label in model.labels:
        score=model.get_score(question,label)
        if score>max_score:
            max_score=score
            best_label=label
    return best_label
# These are the functions you need to implement
def calculate_accuracy(confusion_matrix, labels):
    """Returns the accuracy based on the contents of a confusion matrix

    Keyword arguments:
    confusion_matrix -- a dictionary, as defined in the Confusion Matrix question
    labels -- a set of strings, all the possible labels
    """
    # TODO
    # count,total=0,0
    # for label, freq in confusion_matrix:
    #     if(label[0]==label[1]):
    #         count+=freq
    #     total+=freq    
    
    tp = sum(confusion_matrix.get((label, label), 0) for label in labels)
    total = sum(confusion_matrix.values())
    return tp/total

def calculate_precision(confusion_matrix, labels):
    """Returns a dict containing the precision for each label based on the contents of a confusion matrix

    Keyword arguments:
    confusion_matrix -- a dictionary, as defined in the Confusion Matrix question
    labels -- a set of strings, all the possible labels
    """
    # TODO
    precision_dictionary={}
    for label in labels:
        tp = confusion_matrix.get((label,label),0)
        fp = sum(confusion_matrix.get((other, label), 0) for other in labels if other!=label)
        precision_dictionary[label]=tp/(tp+fp) if (tp + fp) > 0 else 0 
    return precision_dictionary

def calculate_recall(confusion_matrix, labels):
    """Returns a dict containing the recall for each label based on the contents of a confusion matrix

    Keyword arguments:
    confusion_matrix -- a dictionary, as defined in the Confusion Matrix question
    labels -- a set of strings, all the possible labels
    """
    # TODO
    recall_dictionary={}
    for label in labels:
        tp = confusion_matrix.get((label,label),0)
        fn = sum(confusion_matrix.get((label,other), 0) for other in labels if other!=label)
        recall_dictionary[label]=tp/(tp+fn) if (tp + fn) > 0 else 0 
    return recall_dictionary

def calculate_macro_f1(confusion_matrix, labels):
    """Returns the Macro F-Score based on the contents of a confusion matrix

    Keyword arguments:
    confusion_matrix -- a dictionary, as defined in the Confusion Matrix question
    labels -- a set of strings, all the possible labels
    """
    precision = calculate_precision(confusion_matrix, labels)
    recall = calculate_recall(confusion_matrix, labels)

    f1=[]
    for label in labels:
        p=precision[label];r=recall[label]
        f1.append(2*p*r/(p+r)) if (p + r) > 0 else 0 
    return sum(f1)/len(f1) if len(f1)>0 else 0

def main(filename, iterations, read_data, model_maker, learn, find_best_code, get_confusion_matrix, calculate_accuracy, calculate_macro_f1):
    """Trains and evaluates a model on some read_data

    Keyword arguments:
    filename -- a string, the location of a json file containing data
    iterations -- an integer, the number of iterations of training to do
    read_data -- a function, as defined in the Data question
    model_maker -- a class, as defined in the Model question
    learn -- a function, as defined in the Learning question
    find_best_code -- a function, as defined in the Inference question
    get_confusion_matrix -- a function, as defined in the Confusion Matrix question
    calculate_accuracy -- a function, as defined in the Evaluation Metrics question
    calculate_macro_f1 -- a function, as defined in the Evaluation Metrics question
    """
    # TODO
    keys,labels = read_data(filename)
    model = model_maker(labels, keys["train"])
    
    dev_scores=[]
    for _ in range(iterations):
        for quest, label in keys["train"]:
            learn(quest,label,model,find_best_code)
        cm = get_confusion_matrix(keys["dev"], model, find_best_code)
        dev_scores.append({"accuracy":calculate_accuracy(cm,labels),
        "macro-f1":calculate_macro_f1(cm,labels)})
    
    test_score={}
    cm = get_confusion_matrix(keys["test"], model, find_best_code)
    test_scores={"accuracy" : calculate_accuracy(cm,labels),
    "macro-f1" : calculate_macro_f1(cm,labels)}

    return dev_scores, test_scores

if __name__=="__main__":
    dev_scores,test_scores=main("a2_data.json",10,read_data,CodeModel,learn,find_best_code,get_confusion_matrix,calculate_accuracy,calculate_macro_f1)
    print(dev_scores)
    print(test_scores)
