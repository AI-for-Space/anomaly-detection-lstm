from xml.etree.ElementPath import prepare_child
import numpy as np
import pandas as pd
import math
import time        
import random
import scipy.stats as stats

# Definicion de la clase OeSNN-UAD

class neuron():
    ID = 0
        
    PSP_max = 0.0
    gamma = 0.0
    outputValue = 0.0
    M = 0.0
    PSP = 0.0
    additionalTime = 0.0
    
    def __init__(self, ID=0, s_weights= [], PSP_max = 0.0, gamma=0.0, M=0.0, PSP=0.0, additionalTime=0.0):
        self.ID = ID
        self.s_weights = s_weights
        self.PSP_max = PSP_max
        self.gamma = gamma
        self.M = M
        self.PSP = PSP
        self.outputValue = 0.0
        self.additionalTime = additionalTime
    
class inputValue():
    timeStamp = ""
    value = 0.0
    r_label = False
    
    def __init__(self, timeStamp="", value=0.0, r_label=False):
        self.timeStamp = timeStamp
        self.value = value
        self.r_label = r_label
        
class GRF():
    mu = 0.0
    sigma = 0.0
    exc = 0.0
    
    def __init__(self, mu=0.0, sigma=0.0, exc=0.0):
        self.mu = mu
        self.sigma = sigma
        self.exc = exc
        
        
class ConfucionMatrix():
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    def __init__(self, TP, FP, FN, TN):
        self.TP = TP
        self.FP = FP
        self.FN = FN
        self.TN = TN
        
        
class data_auc():
    label = False
    value = False
    
    def __init__(self, label, value):
        self.label = label
        self.value = value
        
class inputNeuron():
    ID = 0
    firingTime = 0.0
    
    def __init__(self, ID, firingTime):
        self.ID = ID
        self.firingTime = firingTime
        
# Definiciones de variables globales
global CNOsize
global Wsize
global NOsize
global Beta
global NIsize
global TS
global sim
global C
global mod
global ErrorFactor
global AnomalyFactor


global OutputNeurons #Pointers to output neurons (outputneuron repository)
global X #input dataset
global Y #predicted values of eSNN
global U #classification of each X[t] input value
global E #error between predicted Y[t] and X[t]
global GRFs #input GRFs
global spikeOrder #firing order of input neurons for current X[t]

global datasetSize;

global ConfusionMatrix #Confusion matrix

global neuronAge

global threshold
threshold = 0.0

def ClearStructure():
    global OutputNeurons
    global X
    global Y
    global U
    global E
    global GRFs
    global spikeOrder
    global NeuronFired
    global neuronAge
    global ConfusionMatrix
    
    OutputNeurons = np.array([])
    X = np.array([])
    Y = np.array([])
    U = np.array([], dtype=np.bool8)
    E = np.array([])
    GRFs = np.array([])
    spikeOrder = []
    NeuronFired = np.array([])
    neuronAge = 0;
    ConfusionMatrix = ConfucionMatrix(0, 0, 0, 0)
    
## Algoritmo 2
def InitializeGRFs(Windows):
    global GRFs
    global Beta
    global NIsize
    global GRFs
    global spikeOrder
    global TS
    
    inputNeurons = np.array([])
    
    I_max = np.max(Windows)
    I_min = np.min(Windows)
    
    # TODO:
    # Se puede optimizar con un map
    for i in range(0, len(GRFs)):
        
        for _ in range(0,NIsize):
            # Formula 1 y 2
            GRFs[i].mu = I_min + ((2.0 * i - 3.0) / 2.0) * ((I_max - I_min) / (float(NIsize) - 2))
            GRFs[i].sigma = (1.0 / Beta) * (((I_max - I_min) / (float(NIsize) - 2)));
            
            if GRFs[i].sigma == 0.0:
                GRFs[i].sigma = 0.5
            
            # Formula 3
            exc = (math.exp(-0.5 * pow(((Windows[len(Windows) - 1] - GRFs[i].mu) / GRFs[i].sigma), 2)))
            # Formula 4
            firingTime = (TS * (1 - exc))
            newIN = inputNeuron(i, firingTime)
            inputNeurons = np.append( inputNeurons, newIN)
            
            aux = sorted(inputNeurons, key=lambda x: x.firingTime)
            spikeOrder.append(aux)

    
def CalculateAvgW(vec): # average of values in vec
    return np.mean(vec)

def CalculateStdW(vec, avg=0.0): # standard deviation of values in vec
    return np.std(vec)

def CalculateDistance(x, y):
    return np.linalg.norm(np.array(x)-np.array([y]))


## Algorimto 3 Inicializacion de InitializeNeuron

def InitializeNeuron( Window:np.array([])): #Initalize new neron n_i
    global NIsize
    global mod
    global C
    global neuronAge
    
    neuron_i = neuron()

    neuron_i.s_weights = [0 for i in range(NIsize)]
    
    order = 0
    for k in range(NIsize):
        for j in range(len(spikeOrder[k])):
            neuron_i.s_weights[spikeOrder[k][j].ID] += pow(mod,order)
            order += 1

    order = 0
    for k in range(NIsize):
        for j in range(len(spikeOrder[k])):
            neuron_i.PSP_max += neuron_i.s_weights[spikeOrder[k][j].ID] * pow(mod,order)
            order += 1
    
    neuron_i.gamma = neuron_i.PSP_max * C
 
    neuron_i.outputValue = random.uniform(CalculateAvgW(Window), CalculateStdW(Window))#np.random.normal(CalculateAvgW(Window), CalculateStdW(Window), NIsize)
    neuron_i.M += 1
    neuron_i.additionalTime = neuronAge
    neuronAge += 1
    
    return neuron_i

## Algoritmo 4
def FindMostSimilar(neuron_i: neuron):
    global OutputNeurons
    
    minDist = CalculateDistance(neuron_i.s_weights, OutputNeurons[0].s_weights)
    minIdx = 0
    
    if len(OutputNeurons) > 1:
        for k in range(1, len(OutputNeurons)):
            dist = CalculateDistance(neuron_i.s_weights, OutputNeurons[k].s_weights)
            if dist < minDist:
                minDist = dist
                minIdx = k
                
    return OutputNeurons[minIdx]
    
## Algoritmo 5
def UpdateNeuron(neuron_i: neuron, neuron_s: neuron): # Update neuron n_s in output repository
    
    for j in range(len(neuron_s.s_weights)):
        neuron_s.s_weights[j] = (neuron_i.s_weights[j] + neuron_i.s_weights[j] * neuron_s.M) / (neuron_s.M + 1)
        
    neuron_s.gamma = (neuron_i.gamma + neuron_i.gamma * neuron_s.M) / (neuron_s.M + 1)
    neuron_s.outputValue = (neuron_i.outputValue + neuron_i.outputValue * neuron_s.M) / (neuron_s.M + 1)
    neuron_s.additionalTime = (neuron_i.additionalTime + neuron_i.additionalTime * neuron_s.M) / (neuron_s.M + 1)
    neuron_s.M += 1
    
    del neuron_i
    
## Algoritmo 7
def ErrorCorrection(neuron_f: neuron, x: float):
    global ErrorFactor
    
    neuron_f.outputValue += (x - neuron_f.outputValue) * ErrorFactor
    return neuron_f
    

## Algoritmo 8 Clasificacion
def ClassifyAnomaly(x:float, y: float):
    global E
    global Wsize
    global U
    global AnomalyFactor
        
    eVec = np.array([])
    
    if len(E) >= Wsize:
        for k in range(len(E)-2, len(E)-2- (Wsize-2), -1):
            if U[k] == False:
                eVec = np.append(eVec, E[k])

    else:
        for k in range(len(E) - 1):
            if U[k] == False:
                eVec = np.append(eVec, E[k])
              
    ret = (E[-1] - np.mean(eVec)) > AnomalyFactor * np.std(eVec)

    return ret

def ReplaceOdest(neuron_i:neuron):
    global OutputNeurons
    
    oldest = OutputNeurons[0].additionalTime
    oldestIdx = 0
    
    for k in range(1, len(OutputNeurons)):
        if OutputNeurons[k].additionalTime < oldest:
            oldest = OutputNeurons[k].additionalTime
            oldestIdx = k
            
    OutputNeurons[oldestIdx] = neuron_i
    

        
## Algoritmo 6
def NeuronSpikeFirst():
    global OutputNeurons
    global mod
    global NIsize
    
    order = 0
    maxCurrentPSPDiff = 0.0
    maxCurrentPSPDiffIdx = -1
    
    for k in range(len(OutputNeurons)):
        OutputNeurons[k].PSP = 0.0
    
    toFire = np.array([], dtype=int)
    
    for j in range(NIsize):
        for l in range(len(spikeOrder[j])):
            for i in range(len(OutputNeurons)):
                OutputNeurons[i].PSP += pow(mod,order) * OutputNeurons[i].s_weights[spikeOrder[j][l].ID]
            
            for i in range(len(OutputNeurons)):
                if OutputNeurons[i].PSP > OutputNeurons[i].gamma:
                    toFire = np.append(toFire,i)
                    
            maxCurrentPSPDiff = 0.0
            maxCurrentPSPDiffIdx = -1           
            
            if toFire.size > 0:
                for i in range(len(toFire)):
                    if (OutputNeurons[toFire[i]].PSP  - OutputNeurons[toFire[i]].gamma) > maxCurrentPSPDiff:
                        maxCurrentPSPDiff = OutputNeurons[toFire[i]].PSP - OutputNeurons[toFire[i]].gamma;
                        maxCurrentPSPDiffIdx = toFire[i]
                        
                toFire = np.array([])
                return OutputNeurons[maxCurrentPSPDiffIdx]
                
            order+=1
    return None
            
            
def CalculateMaxDist():
    global NIsize
    global mod
    
    v1 = np.array([], dtype=float)
    v2 = np.array([], dtype=float)
    
    for i in range(NIsize):
        v1 = np.append(v1 , (pow(mod, NIsize - 1 - i)))
        v2 = np.append(v2 , pow(mod, i))
        
    return np.linalg.norm(v1 - v2)


def Train():
    
    global CNOsize
    global Wsize
    global NOsize
    global Beta
    global NIsize
    global TS
    global sim
    global C
    global mod
    global ErrorFactor
    global AnomalyFactor

    global OutputNeurons 
    global X
    global Y
    global U
    global E
    global GRFs 
    global spikeOrder 
    global NeuronFired

    global datasetSize;
    
    Dmax = CalculateMaxDist()
    
    CNOsize = 0
    for j in range(NIsize):
        newGRF = GRF()
        GRFs = np.append(GRFs, newGRF)
        
    Window = np.array([], dtype=float)
    
    for k in range(Wsize):
        Window = np.append(Window, X[k][1])

    
    random.seed()
    
    avgW = CalculateAvgW(Window);
    stdW = CalculateStdW(Window, avgW);
    #Y = [random.gauss(avgW, stdW) for _ in range(Wsize)]
    Y = np.random.normal(avgW, stdW, Wsize)
    
    
    for k in range(Wsize):
        E = np.append(E, abs(X[k][1] - Y[k]))
        NeuronFired = np.append(NeuronFired, -1)
        
    for k in range(Wsize):
        U = np.append(U, False)
        
    for t in range(Wsize, len(X)):
        Window = np.delete(Window, 0)
        Window = np.append(Window, X[t][1])
        
        InitializeGRFs(Windows=Window)        
        neuron_i = InitializeNeuron(Window)
        
        neuron_s = neuron()
        if CNOsize>0:
            neuron_s = FindMostSimilar(neuron_i= neuron_i)
            
        if CNOsize >0 and CalculateDistance(neuron_i.s_weights, neuron_s.s_weights) <= sim* Dmax:
            UpdateNeuron(neuron_i= neuron_i, neuron_s= neuron_s)
        elif CNOsize < NOsize:
            OutputNeurons = np.append(OutputNeurons, neuron_i)
            CNOsize+=1
        else:
            ReplaceOdest(neuron_i)
            
        
        ##################
        
        n_f = NeuronSpikeFirst()
        
        NeuronFired = np.append(NeuronFired, n_f.additionalTime)
        y_t_1 = n_f.outputValue
        
        Y = np.append(Y, y_t_1)
        x_t_1 = X[t][1]
        e_t_1 = abs(x_t_1 - y_t_1)
        E = np.append(E, e_t_1)
        
        u_t_1 = ClassifyAnomaly(x_t_1, y_t_1)
        
        U = np.append(U, u_t_1)
        if u_t_1  == False: 
            n_f = ErrorCorrection(n_f,x_t_1)        
            
    #Desnormalizamos los valores de Y
    Y = (Y * (np.max(X[:,3]) - np.min(X[:,3])) + np.min(X[:,3]))
    print("Training finished")
            
def loadData(file: str = None):
    global X 
    tmp = []
    if file is None:
        for i in range(1000):
            tmp.append([i, (np.random.rand())/11, 0])
            
        
        for i in range(100):
            tmp.append([i+1000, (np.random.rand()+10)/11, 0])
        X = np.array(tmp)
    else:      
         
        df = pd.read_csv(file, sep=',', header=None, skiprows=1, names=['timestamp', 'value', 'r_label'])
        df['r_label'] = df['r_label'].fillna(0).astype(bool)
        df['timestamp'] = pd.to_datetime(df['timestamp']) # df['timestamp'].values.astype(np.int64) // 10 ** 9#[x+300 for x in range(0, len(df['timestamp']))]
        df['timestamp'] = df['timestamp'].values.astype(np.int64) // 10 ** 9
        df['timestamp'] = df['timestamp'] - df['timestamp'].min()
        df['value']= df['value'].astype(float)
        
        #print(df.info())
        
        df['rawData'] = df.loc[:, 'value']
        
        # normalize dataframe column value to [0,1]
        column = 'value'
        df[column] = (df[column] - df[column].min()) / (df[column].max()-df[column].min())
        
        X = df.to_numpy()
        
                
def CalculateConfusionMatrix():
    global X
    global U
    global ConfusionMatrix
    
    for t in range(len(X)):
        if X[t][2] == U[t]:
            if X[t][2] == True and U[t] == True:
                ConfusionMatrix.TP+=1
            elif X[t][2] == False and U[t] == False:
                ConfusionMatrix.TN+=1
        else:
            if X[t][2] == True and U[t] == False:
                ConfusionMatrix.FN+=1
            elif X[t][2] == True and U[t] == False:
                ConfusionMatrix.FP+=1
            
   
def CalculatePrecission():
    global ConfusionMatrix
    
    if ConfusionMatrix.TP == 0: return 0.0000001 # Perfecto
    return ConfusionMatrix.TP / (ConfusionMatrix.TN + ConfusionMatrix.FP)
    

def CalculateRecall():
    global ConfusionMatrix
    
    if ConfusionMatrix.TP == 0: return 0.0000001 # Perfecto
    return ConfusionMatrix.TP / (ConfusionMatrix.TN + ConfusionMatrix.FN)

def CalculateF_Measure(precision = 0.0, recall = 0.0):
    return (2 * precision * recall) / (precision + recall)

def saveResults(file: str = "C:/Users/Jorge/Documents/Github/TFM/results.csv"):
    maxValue =  np.max(X[:,3])
    precision = CalculatePrecission()
    recall = CalculateRecall()
    fmeasure = CalculateF_Measure(precision, recall)
    with open(file, 'w') as f:
        f.write("timestamp,value,predicted_value,error,p_label,fired,r_label,fmeasure\n")
        for i in range(len(X)):
            f.write(str(int(X[i][0])) + "," + str(float(X[i][3])) + "," + str(Y[i]) + "," + str(abs(float(X[i][3]) - (Y[i]))) +"," + str(U[i]) +"," + str(NeuronFired[i]) +"," + str(bool(X[i][2])) + "," + str(fmeasure) + "\n",)
            
        f.close()

def OeSNNUAD(NO_size = 50,    
         W_size = 500,
         NI_size = 10,    
         _Beta = 1.6,
         _TS = 1000,    
         _sim = 0.15,
         _mod = 0.6,    
         _C = 0.6,
         _ErrorFactor = 0.9,
         _AnomalyFactor = 3,
         file_input = "",
         file_output =""):
    
    global CNOsize
    global Wsize
    global NOsize
    global Beta
    global NIsize
    global TS
    global sim
    global C
    global mod
    global ErrorFactor
    global AnomalyFactor

    global OutputNeurons 
    global X
    global Y
    global U
    global E
    global GRFs 
    global spikeOrder 
    
    Wsize = W_size
    NOsize = NO_size
    Beta = _Beta
    NIsize = NI_size
    TS = _TS
    sim = _sim
    C = _C
    mod = _mod
    ErrorFactor = _ErrorFactor
    AnomalyFactor = _AnomalyFactor
    
    
    
    global threshold
    ClearStructure()

    

    loadData(file_input)
    Train()
    CalculateConfusionMatrix()
    
    precision = CalculatePrecission()
    recall = CalculateRecall()
    fmeasure = CalculateF_Measure(precision, recall)
    
    if fmeasure > threshold:
        threshold = fmeasure
        maxFMeasuer = fmeasure
        maxPrecision = precision
        maxRecall = recall
        
        saveResults(file_output)
        
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-Score: ", fmeasure)
    

    
    ClearStructure()
                      