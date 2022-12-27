import torch


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor, cleanframe):    
        self.shape = input_tensor.size()
        if not cleanframe is None:
            #print("====")
            #print(input_tensor)
            #print(cleanframe)
            output = torch.zeros_like(input_tensor.mean(dim=self.dim, keepdim=True))
            if self.consensus_type == 'avg':

                for i in range(self.shape[0]):
                    assert cleanframe[i].sum() > 0
                    
                    output[i] = input_tensor[i][cleanframe[i]].mean(dim=0,keepdim=True)

            else:
                raise NotImplementedError
            #print(output)
        else:
            if self.consensus_type == 'avg':
                output = input_tensor.mean(dim=self.dim, keepdim=True)
            elif self.consensus_type == 'identity':
                output = input_tensor
            elif self.consensus_type == 'max':
                output, _ = input_tensor.max(dim=self.dim, keepdim=True)
            else:
                output = None

        return output


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input, cleanframe = None):
        return SegmentConsensus(self.consensus_type, self.dim)(input, cleanframe)
