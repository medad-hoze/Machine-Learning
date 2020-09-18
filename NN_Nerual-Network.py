
import numpy as np


#  Activation Functions

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

def relu(x):
    return max(x,0)

def relu_derivative(x):
    return 1 if x>0 else 0



class Nerual_Network():
    
    def __init__(self,x,y,epoch):
        self.x        = x
        self.y        = y
        self.shapeX   = x.shape
        self.shapeY   = y.shape
        self.epoch    = epoch
        self.Pred     = []

        
    def Weights_Bias(self,Dimensies):
        return [2 * np.random.random(Dimensies) - 1,np.zeros(Dimensies[1])]
        
    def Build_NetWork(self,layer,w,b):
        
        l = sigmoid(np.dot(layer,w)+b)
        return l
        
    def Loss(self,x,num):
        if num%10000 == 0:
            print (np.sum(x**2))
        
                
    def ReValue_Weight_Bias(self,layer,delta,w,b):
        w += np.dot(layer.T,delta)
        b += delta.sum(axis=0)
        
        return w,b
    
    def ReValue_Last_Weight_Bias(self,layer,delta,w,b):
        
        w += np.dot(layer.T,delta)
        b += delta.sum()
        
        return w,b
            
    def ReValue_Error_Delta(self,layer,Upper_delta,weight):
        
        error = np.dot(Upper_delta,weight.T)
        delta = error * sigmoid_derivative(layer)
        
        return error,delta
    
    def ReValue_Last_Error_Delta(self,Out_put,layer):
            error= Out_put-layer.T
            delta = error.T * sigmoid_derivative(layer)
            
            return error,delta
    
    def Fit(self):
        
        w0,b0  = self.Weights_Bias((self.shapeX[1],4))
        w1,b1  = self.Weights_Bias((4,8))
        w2,b2  = self.Weights_Bias((8,4))
        w3,b3  = self.Weights_Bias((4,1))
        

        for i in range(self.epoch):
            
            l1 = self.Build_NetWork(x,w0,b0)
            l2 = self.Build_NetWork(l1,w1,b1)
            l3 = self.Build_NetWork(l2,w2,b2)
            l4 = self.Build_NetWork(l3,w3,b3)
            

            l4_error,l4_delta = self.ReValue_Last_Error_Delta(y,l4)
            
            l3_error,l3_delta = self.ReValue_Error_Delta(l3,l4_delta,w3)
            l2_error,l2_delta = self.ReValue_Error_Delta(l2,l3_delta,w2)
            l1_error,l1_delta = self.ReValue_Error_Delta(l1,l2_delta,w1)
    

            w0,b0 = self.ReValue_Weight_Bias     (x,l1_delta,w0,b0)
            w1,b1 = self.ReValue_Weight_Bias     (l1,l2_delta,w1,b1)
            w2,b2 = self.ReValue_Weight_Bias     (l2,l3_delta,w2,b2)
            w3,b3 = self.ReValue_Last_Weight_Bias(l3,l4_delta,w3,b3)
            
            
            self.Loss(l4_error,i)
        
        self.Pred.append([[w0,b0],[w1,b1],[w2,b2],[w3,b3]])
            
        return l4_delta
    
    def Predict(self,x):
        
            w0,b0 = self.Pred[0]
            w1,b1 = self.Pred[1]
            w2,b2 = self.Pred[2]
            w3,b3 = self.Pred[3]
        
            l1 = self.Build_NetWork(x,w0,b0)
            l2 = self.Build_NetWork(l1,w1,b1)
            l3 = self.Build_NetWork(l2,w2,b2)
            l4 = self.Build_NetWork(l3,w3,b3)
            
            l4_error,l4_delta = self.ReValue_Last_Error_Delta(y,l4)
            
            return l4
    
    
    
    
np.random.seed(1)

x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],[1,1,1]])
y = np.array([0,0,1,1,1])

        
nn1 = Nerual_Network(x,y,1000000)

nn1.Fit()


x = np.array([[1,1,1],[1,0,1],[1,0,1],[0,1,1],[0,0,1]])


print (nn1.Predict(x))



        


