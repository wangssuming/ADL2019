import torch


class ExampleNet(torch.nn.Module):
    """

    Args:

    """



    def __init__(self, dim_embeddings,
                 similarity='inner_product'):
        super(ExampleNet, self).__init__()
        
        self.gru = torch.nn.GRU(dim_embeddings, 108, bidirectional=True, batch_first=True)
        
        self.s = torch.nn.Softmax(dim=1)

    def forward(self, context, context_lens, options, option_lens):
        print(context.size())
#        print(options)
        context_out, context_hidden = self.gru(context)
        #context_out = self.mlp(context_out).max(1)[0]
        context_h = context_hidden.transpose(1,0)
        context_h = context_h.contiguous().view(context.size(0),-1)
        #print(context_out.shape)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option_out, option_hidden = self.gru(option)
            option_h = option_hidden.transpose(1,0)
            option_h = option_h.contiguous().view(context.size(0),-1)
            #option_out = self.mlp(option_out).max(1)[0]
            #logit = ((context_out - option_out) ** 2).sum(-1)    
            logit = torch.nn.CosineSimilarity(dim=1)(context_h, option_h)       
            logits.append(logit)
            #print(logit.size())
        logits = torch.stack(logits, 1)
        #print(logits.size())
#        print(logits.size())
        #return logits
        return self.s(logits)
