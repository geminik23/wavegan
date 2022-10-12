import os
import torch
from torchaudio.datasets import SPEECHCOMMANDS
from glob import glob

class SubsetSC(SPEECHCOMMANDS):
    LABELS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    LABEL2INDEX={}

    def __init__(self, data_path, fixed_size=16000, subset:str="training"):
        super().__init__(data_path, download=True, subset=subset)
        self.fixed_size = fixed_size

        if subset == "validation":
            self._walker = self._load_digit("validation_list.txt")
        elif subset == "testing":
            self._walker = self._load_digit("testing_list.txt")
        elif subset == "training":
            e = self._load_digit("validation_list.txt") + self._load_digit("testing_list.txt")
            e = set(e)
            
            files = []
            for l in SubsetSC.LABELS:
                files += glob(os.path.join(self._path, l) + "/*.wav")
            self._walker = [f for f in files if f not in e]
    
    def _load_digit(self, filename):
        filepath = os.path.join(self._path, filename)
        with open(filepath) as f:
            return [os.path.normpath(os.path.join(self._path, line.strip())) for line in f if line.split('/')[0] in self.LABELS]

    def __getitem__(self, n: int): 
        tensor, _, label, *_ = super().__getitem__(n)
        if tensor.size(-1) != self.fixed_size:
            tensor = torch.nn.functional.pad(tensor, (0, self.fixed_size-tensor.size(-1)))
        return tensor, self.LABEL2INDEX[label]


SubsetSC.LABEL2INDEX = {l:i for i, l in enumerate(SubsetSC.LABELS)}

def get_speechcommand_dataset(config, fixed_size=16000): # 1 secs
    sc_test = SubsetSC( config.dataset_path, fixed_size=fixed_size, subset="testing")
    sc_train = SubsetSC( config.dataset_path, fixed_size=fixed_size, subset="training")
    return sc_train, sc_test
