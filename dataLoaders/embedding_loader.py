from torch.utils.data import Dataset
import torch
import os
import pickle

INTENSITY_STD = 23.12

class EmbeddingDataset(Dataset):
    """Dataset of Hurricane Embeddings from all the given years"""

    def __init__(self, dataset_dir, years):
        """
        Args:
            dataset_dir (string): Path to the dataset file.
            years (list[int]): the years from which to get the embeddings
        """

        self.dataset_path = dataset_dir
        self.years = years
        
        self.sample_dict = {}

        with open(os.path.join(self.dataset_path), 'rb') as f:
            loaded_dict = pickle.load(f)
            
            for y in years:
                self.sample_dict[y] = loaded_dict[y]

        self.embedding_tuples = []

        for y in years:
            for h in self.sample_dict[y]:
                for i in range(h[0].shape[0] - 2):
                    
                    x_1, x_2, x_3 = h[0][i].detach(), h[0][i + 1].detach(), h[0][i + 2].detach()
                    y_1, y_2, y_3 = h[1][i].detach(), h[1][i + 1].detach(), h[1][i + 2].detach()
                    
                    x = torch.cat([x_1, y_1, x_2, y_2])
                    y = torch.cat([x_3, y_3])
                    
                    self.embedding_tuples.append((x, y))
            

    def __len__(self):
        """ Returns number of 3-tuples of embeddings and intensities. """
        
        return len(self.embedding_tuples)

    def __getitem__(self, idx):
        """ Returns a pair of (input, output) for the Oracle-PHURIE model. """
        
        x, y = self.embedding_tuples[idx]
        return x, y
    
    def get_hurricanes(self):
        """ Generator which returns tuples of (hurricane embeddings, hurricane intensities). """
        
        for y in self.years:
            for h in self.sample_dict[y]:
                yield h[0].detach(), h[1].detach()