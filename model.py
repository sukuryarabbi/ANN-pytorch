from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self,input_dim = 4,output_dim=3):
        super().__init__()
        self.model = nn.Sequential(
            self.make_NN_block(input_dim, input_dim*2),
            self.make_NN_block(input_dim*2, input_dim*4),
            self.make_NN_block(input_dim*4, output_dim, final_layer=True))
    
    def make_NN_block(self,input_dim,output_dim,final_layer = False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_dim,output_dim),
                nn.ReLU())
        else:
            return nn.Sequential(
                nn.Linear(input_dim,output_dim))
    
    def forward(self,x):
        x = self.model(x)
        
        return x
        