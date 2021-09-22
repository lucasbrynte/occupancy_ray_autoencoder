"""
Autoencoder of cylindrical signals, where the code is low-dimensional and coursely smapled on the cylinder.
In our case, the cylindrical input signal consists of dense latent codes of the (outer) occlusion ray autoencoder.
"""


import torch




class CylinderNet(torch.nn.Module):
    
    def __init__ (self, layer_structure, sample_structure, kernel_size):
        
        """
          layer_structure decides the channel sizes of each layer.
          sample_structure decides how we subsample
        """
        
        
        super(CylinderNet,self).__init__()
        
        # We pad the image in angle-direction circularly, and use a manual padder to pad 
        # with zeros in z-direction
        self.layers = torch.nn.ModuleList(
                torch.nn.Conv2d(layer_structure[k],layer_structure[k+1],kernel_size=kernel_size,
                               padding = (0,kernel_size-1), padding_mode = 'circular')
                for k in range(len(layer_structure)-2)
            )
        
        self.padder = torch.nn.ConstantPad2d((0,0,kernel_size-1,kernel_size-1),0)
        
        # we want to apply the last layer outside of a loop, therefore define it separately
        self.lastlayer = torch.nn.Conv2d(layer_structure[-2],layer_structure[-1],kernel_size=kernel_size,
                               padding = (0,kernel_size-1), padding_mode = 'circular')
        
        # using convenience function from pytorch for downsampling, uses bilinear interpolation
        self.samplers = torch.nn.ModuleList(
            torch.nn.AdaptiveAvgPool2d(sample_structure[k])
            for k in range(len(sample_structure)))
        
        self.ReLU=torch.nn.ReLU()
        
        
        
    def forward(self,x):
        
        for k,layer in enumerate(self.layers):
            x=self.padder(x)
            x=layer(x)
            x=self.samplers[k](x)
            x=self.ReLU(x)

           
        x=self.padder(x)   
        x=self.lastlayer(x)
        x=self.samplers[-1](x)
        
        return x
    


