import sys, os
sys.path.append('../../hyena-dna')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                
import torch
from einops import rearrange, repeat
from scgenehyena.model import ScGeneHyena
from scgenehyena.utils import get_toy_data
import unittest



class TestScGeneHyena(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        
        # random seed
        torch.manual_seed(42)
        
        # parameters
        cls.num_genes = 20000
        cls.num_cells = 4
        cls.d_model = 64
        cls.depth = 2
        cls.num_blocks = 4

        # create model
        cls.model = ScGeneHyena(
            num_genes=cls.num_genes,
            dim=cls.d_model,
            depth=cls.depth,
            max_length=32768,
            pretrain_head='masked_reconstruction',
            num_blocks=cls.num_blocks,
        )
        
        # input data without masking
        cls.exp_input = get_toy_data(cls.num_genes, cls.num_cells)

        # input data with masking
        cls.masked_input, cls.mask = get_toy_data(cls.num_genes, cls.num_cells, mask_ratio=0.15)


    def test_model(self):
    
        """
        test basic model function
        """
        
        print("\n\n\n=== Testing basic model function ===")
        
        reconstruction, cell_state = self.model(self.exp_input)
        
        print(f"Input shape: {self.exp_input.shape}")
        print(f"Reconstruction output shape: {reconstruction.shape} (should be {self.num_cells}, {self.num_genes}, 1)")
        print(f"Cell state shape: {cell_state.shape} (should be {self.num_cells}, 4)")
        
        assert reconstruction.shape == (self.num_cells, self.num_genes, 1), "Reconstruction shape mismatch!"
        assert cell_state.shape == (self.num_cells, self.num_cells), "Cell state shape mismatch!"   
        

    def test_masked(self):
    
        """
        test masked pretraining
        """
         
        print("\n\n\n=== Testing masked pretraining ===")  
        
        masked_reconstruction, _ = self.model(self.masked_input)
        loss = torch.nn.MSELoss()(masked_reconstruction[~self.mask], 
                            self.masked_input[~self.mask])
        
        print(f"Masked reconstruction loss: {loss.item():.4f}")
        
        assert not torch.isnan(loss), "Loss is NaN!"

    
    def test_gradients(self):
        
        """
        test gradients
        """
        
        print("\n\n\n=== Testing gradients ===")
        
        self.model.zero_grad()
        
        reconstruction, _ = self.model(self.exp_input)
        loss = torch.nn.MSELoss()(reconstruction, self.exp_input)
        loss.backward()
        
        # check gradients for all parameters
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                assert not torch.all(param.grad == 0), f"{name} has zero gradients!"
                assert not torch.isnan(param.grad).any(), f"{name} has NaN gradients!"
                assert param.grad.abs().max() < 1e5, f"{name} has exploding gradients!"

            else:
                print(f"Warning: No gradients for {name}")

    
    def test_output(self):
    
        """
        test output range
        """
        
        masked_reconstruction, _ = self.model(self.masked_input)
        
        print("\n\n\n=== Testing output range ===")
        print(f"Reconstruction min: {masked_reconstruction.min().item():.4f}")
        print(f"Reconstruction max: {masked_reconstruction.max().item():.4f}")
        assert 0 <= masked_reconstruction.min() <= masked_reconstruction.max() <= 1, "Reconstruction should be in [0,1] for sigmoid output"


        
if __name__ == "__main__":
    unittest.main(verbosity=2)







    
