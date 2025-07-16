import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,num_classes=150):
        super(CNN, self).__init__()

        def cnn_layer(input_features,output_features,ker_size=3,stride=1,pad=0,neg_slope=0):
            return nn.Sequential(nn.Conv2d(input_features,output_features,ker_size,stride,pad),
                                 nn.BatchNorm2d(output_features),
                                 nn.LeakyReLU(negative_slope=neg_slope),
                                 nn.MaxPool2d(2,2))
        
        def linear_layer(inp,out,neg_slop=0,p=0.5):
            return nn.Sequential(nn.Linear(inp,out),
                                 nn.BatchNorm1d(out),
                                 nn.LeakyReLU(negative_slope=neg_slop),
                                 nn.Dropout(p))
        
        self.cnn = nn.Sequential(cnn_layer(3,16,5),
                                 cnn_layer(16,32,5),
                                 cnn_layer(32,64,5))
        
        self.fc = nn.Sequential(nn.Flatten(),
                                linear_layer(64*12*12,1024),
                                linear_layer(1024,512),
                                linear_layer(512,256),
                                nn.Linear(256,num_classes))
        

    def forward(self,x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
