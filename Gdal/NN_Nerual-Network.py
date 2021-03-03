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
        self.x        = np.array(x)
        self.y        = np.array(y)
        self.shapeX   = self.x.shape
        self.shapeY   = self.y.shape
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
        
        self.w0,self.b0  = self.Weights_Bias((self.shapeX[1],4))
        self.w1,self.b1  = self.Weights_Bias((4,8))
        self.w2,self.b2  = self.Weights_Bias((8,4))
        self.w3,self.b3  = self.Weights_Bias((4,1))
        

        for i in range(self.epoch):
            
            l1 = self.Build_NetWork(x,self.w0,self.b0)
            l2 = self.Build_NetWork(l1,self.w1,self.b1)
            l3 = self.Build_NetWork(l2,self.w2,self.b2)
            l4 = self.Build_NetWork(l3,self.w3,self.b3)
            

            l4_error,l4_delta = self.ReValue_Last_Error_Delta(y,l4)
            
            l3_error,l3_delta = self.ReValue_Error_Delta(l3,l4_delta,self.w3)
            l2_error,l2_delta = self.ReValue_Error_Delta(l2,l3_delta,self.w2)
            l1_error,l1_delta = self.ReValue_Error_Delta(l1,l2_delta,self.w1)
    

            self.w0,self.b0 = self.ReValue_Weight_Bias     (self.x,l1_delta,self.w0,self.b0)
            self.w1,self.b1 = self.ReValue_Weight_Bias     (l1,l2_delta,self.w1,self.b1)
            self.w2,self.b2 = self.ReValue_Weight_Bias     (l2,l3_delta,self.w2,self.b2)
            self.w3,self.b3 = self.ReValue_Last_Weight_Bias(l3,l4_delta,self.w3,self.b3)
            
            
            self.Loss(l4_error,i)
            
        return l4_delta
    
    def Predict(self,data):
        
            l1 = self.Build_NetWork(data,self.w0,self.b0)
            l2 = self.Build_NetWork(l1,self.w1,self.b1)
            l3 = self.Build_NetWork(l2,self.w2,self.b2)
            l4 = self.Build_NetWork(l3,self.w3,self.b3)
            
            return l4
    
    
# # # # # # # # # #  TEST CODE # # # # # # # # # # # 
            
# Train NetWork
            
np.random.seed(1)

x = [[0,0,1],[0,1,1],[1,0,1],[1,1,1],[1,1,1]]
y = [0,0,1,1,1]

        
nn1 = Nerual_Network(x,y,10000)

nn1.Fit()


# Predict NetWork


data = [[0,0,1],[0,1,1],[1,0,1],[1,1,1],[1,1,1],[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0]]

print (nn1.Predict(data))
