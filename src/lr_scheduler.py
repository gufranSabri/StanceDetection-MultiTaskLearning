import os
import os
from torch.optim.lr_scheduler import _LRScheduler
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, n_epoch, start_decay, last_epoch=-1):
        self.start_decay=start_decay
        self.n_epoch=n_epoch
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = self.last_epoch
        n_epoch=self.n_epoch
        b_lr=self.base_lrs[0]
        start_decay=self.start_decay
        if last_epoch>start_decay:
            lr=b_lr-b_lr/(n_epoch-start_decay)*(last_epoch-start_decay)
        else:
            lr=b_lr
        return [lr]