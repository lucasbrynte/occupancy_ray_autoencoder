"""
Autoencoder of cylindrical signals, where the code is low-dimensional and coursely smapled on the cylinder.
In our case, the cylindrical input signal consists of dense latent codes of the (outer) occlusion ray autoencoder.
"""


import torch

from torch.distributions.multivariate_normal import MultivariateNormal


"""
     class CylinderNet
     
     This is just a CNN defined on a cylinder.
"""

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
                               padding = (0,kernel_size-1), padding_mode = 'circular',bias=False)
        
        # using convenience function from pytorch for downsampling, uses bilinear interpolation
        self.samplers = torch.nn.ModuleList(
            torch.nn.AdaptiveAvgPool2d(sample_structure[k])
            for k in range(len(sample_structure)))
        
        
        #leaky ReLU to avoid dead neurons
        self.ReLU=torch.nn.LeakyReLU()
        
        
        
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
    
    
"""
    class CylEncoder
    
    This net has a CylNet backbone and a final fully connected layer
    
    If the final layer of the CylNet has a spatial dimension k x l,
    the fully connected layer outputs parameters for a multivariate Gaussian
    on R^{k*l}
"""
    
class CylEncoder(torch.nn.Module):
    
    #cylindrical variational autoencoder
    
    def __init__(self, layer_structure,sample_structure,kernel_size):
        
        super(CylEncoder,self).__init__()
        
        # cylindernet to process the input
        
        self.CylNet = CylinderNet(layer_structure,sample_structure,kernel_size)
        
        # fully connected layer to obtain parameters
        # number of parameters
        self.K= sample_structure[-1][0]*sample_structure[-1][1]
        self.nbr_channels = layer_structure[-1]
        
        # for means
        self.fc_mu = torch.nn.Linear(layer_structure[-1]*self.K, layer_structure[-1]*self.K)
        # for covariance_matrices
        self.fc_protocov = torch.nn.Linear(layer_structure[-1]*self.K, layer_structure[-1]*self.K**2)
        
    def forward(self,x):
        
        # process on cylinder
        x = self.CylNet(x)
        
        # get parameters for probability distribution
        mu = self.fc_mu(x.reshape([x.size()[0],-1]))
        A = self.fc_protocov(x.reshape([x.size()[0],-1]))
        
        mu = mu.reshape([-1,self.nbr_channels,self.K])
        A = A.reshape([-1,self.nbr_channels,self.K,self.K])
        
        return mu, A@A.transpose(2,3) # parameters for a multinomial distribution
    
    
"""
 class CylVAE

    A cylindrical autoencoder.
    
    self.Encoder is a CylEncoder
    self.Decoder is a CylNet
    
    Calling forward generates a probability distribution (for generating codewords)
    and one decoded sample
"""


class CylVAE(torch.nn.Module):
    

     def __init__(self, layer_structure,sample_structure,kernel_size):
        
        super(CylVAE,self).__init__()
        
        # Encoder
        
        self.Encoder = CylEncoder(layer_structure,sample_structure[1:],kernel_size)
        
        #
        self.latent_channel_dimension = layer_structure[-1]
        self.latent_spatial_dimension = sample_structure[-1]
        
        # Decoder
        
        layer_structure.reverse()
        sample_structure.reverse()
        self.Decoder = CylinderNet(layer_structure, sample_structure[1:],kernel_size)
        
     def forward(self,x):
        
        #  get parameters from encoder
        
        mu,Sigma = self.Encoder(x)
        
        # define  distribution
        
        q=MultivariateNormal(mu,Sigma)
        
        # sample distribution to get codeword 
        
        z = q.rsample()
        z = z.reshape([-1,self.latent_channel_dimension,
                       self.latent_spatial_dimension[0],self.latent_spatial_dimension[1]])
        
        # process on cylinder
        
        gen = self.Decoder(z)
        return  (q,gen)
    


    

    
