import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.pyplot as plt
import numpy as np

def adjust_score(model, scaler, X_test, classes, targets_test, excludes_test):
    if scaler != None:
        X_test = scaler.transform(X_test)
    # adapt here if the model has multiple outputs. Append '[0]'
    probabs = model.predict(X_test, verbose=0)
   
    score = 0.0
    for i in range(0, probabs.shape[0]):
        classes_sorted = probabs[i].argsort()[::-1]
        result = None
        best = 0
        while result == None:
            temp_result = classes[classes_sorted[best]]
            if temp_result in excludes_test[i]:
                best += 1
            else:
                result = temp_result
        if result == targets_test[i]:
            score += 1.0
    
    score /= len(targets_test)
    print("New adjusted score " + str(score))
    return score

def plot_oracle(config, task, model, scaler, X_test, classes, targets_test, excludes_test):   
    oracle = [0] * len(classes)
    if scaler != None:
        X_test = scaler.transform(X_test)
    # adapt here if the model has multiple outputs. Append '[0]'
    probabs = model.predict(X_test, verbose=0)
    for i in range(0, probabs.shape[0]):
        classes_sorted = probabs[i].argsort()[::-1]
        adjusted_classes_sorted = []
        for j in range(0, min(20, classes_sorted.shape[0])):
            classname = classes[classes_sorted[j]]
            if classname not in excludes_test[i]:
                adjusted_classes_sorted.append(classname)
        best = 0
        while best < len(adjusted_classes_sorted):
            if adjusted_classes_sorted[best] == targets_test[i]:
                oracle[best] += 1
                break
            else:
                best += 1
        
    
    oracle = [sum(oracle[0:i]) for i in range(1, len(oracle)+1)]
    oracle = [x/len(targets_test) for x in oracle]
    
    host = host_subplot(111)
    host.set_xlabel('Ranks')
    host.set_ylabel("Recognition Rate")
    
    host.plot(list(range(0,10)), oracle[0:10], label='Recognition Rate')
    
    plt.title('Oracle')
    plt.savefig(config['base_folder'] + 'classification/oracle_' + task + '.png')
    print("Saving oracle plot to " + config['base_folder'] + 'classification/oracle_' + task + '.png')
    plt.close()
    return oracle

def plot_classification_confidence_histograms(config, task, model, scaler, X_test, classes, targets_test, excludes_test):    
    best_confidence_hist = [0 for i in range(101)]
    true_confidence_hist = [0 for i in range(101)]
    if scaler != None:
        X_test = scaler.transform(X_test)
    probabs = model.predict(X_test, verbose=0)
    for i in range(0, probabs.shape[0]):
        classes_sorted = probabs[i].argsort()[::-1]
        adjusted_classes_and_probabs_sorted = []
        for j in range(0,classes_sorted.shape[0]):
            classname = classes[classes_sorted[j]]
            probab = probabs[i][classes_sorted[j]]
            if classname not in excludes_test[i]:
                adjusted_classes_and_probabs_sorted.append((classname, probab))
        probab = adjusted_classes_and_probabs_sorted[0][1]
        
        try:
            best_confidence_hist[int(round(100*probab))] += 1
        except ValueError:
            return
        best = 0
        while best < len(adjusted_classes_and_probabs_sorted):
            if adjusted_classes_and_probabs_sorted[best][0] == targets_test[i]:
                probab = adjusted_classes_and_probabs_sorted[best][1]
                try:
                    true_confidence_hist[int(round(100*probab))] += 1
                except ValueError:
                    return
                break
            else:
                best += 1
    
    host = host_subplot(111)
    host.set_xlabel('Confidence')
    host.set_ylabel("Probability")
    divisor = sum(true_confidence_hist)
    host.plot(np.array(range(101)), np.array([x/divisor for x in true_confidence_hist]), label='Probability')
    
    plt.title('True Confidence Hist')
    plt.savefig(config['base_folder'] + 'classification/true_confidence_hist_' + task + '.png')
    plt.close()
    print("Saving true confidence histogram to " + config['base_folder'] + 'classification/true_confidence_hist_' + task + '.png')

    host = host_subplot(111)
    host.set_xlabel('Confidence')
    host.set_ylabel("Probability")
    divisor = sum(best_confidence_hist)
    host.plot(np.array(range(101)), np.array([x/divisor for x in best_confidence_hist]), label='Probability')
    
    plt.title('Best Confidence Hist')
    plt.savefig(config['base_folder'] + 'classification/best_confidence_hist_' + task + '.png')
    print("Saving true confidence histogram to " + config['base_folder'] + 'classification/best_confidence_hist_' + task + '.png')   
     
