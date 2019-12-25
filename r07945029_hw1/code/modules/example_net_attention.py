import torch
import numpy as np
import torch.nn.functional as F
torch.cuda.manual_seed_all(9487)

class ExampleNet(torch.nn.Module):
    """

    Args:

    """


    def __init__(self, dim_embeddings,
                 similarity='inner_product'):
        super(ExampleNet, self).__init__()
        self.hidden = 128
        self.rnn1 = torch.nn.GRU(dim_embeddings, self.hidden, 1, bidirectional=True, batch_first=True)
        self.nn = torch.nn.Linear(self.hidden*2, self.hidden*2, bias=False)
        self.attention = torch.nn.Linear(self.hidden*2, 1, bias=False)
        self.rnn2 = torch.nn.GRU(self.hidden*2*4, self.hidden, 1, bidirectional=True, batch_first=True)
        self.similarity = torch.nn.Linear(self.hidden*2*2,1,bias=False)
#        self.projection = torch.nn.Linear(43200,100)

    def forward(self, context, context_lens, options, option_lens):
        context_out, context_hidden = self.rnn1(context)
        context_nn = self.nn(context_out)
#        print(context_nn.size())
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option_out, option_hidden = self.rnn1(option)
            option_nn = self.nn(option_out) 
#            print(option_nn.size())
            C = torch.unsqueeze(context_nn, 1).expand(-1, option.size(1), -1, -1)
            P = torch.unsqueeze(option_nn, 2).expand(-1, -1, context.size(1), -1)
            attention = self.attention(torch.tanh(P + C)).squeeze(dim=-1)
#            print(attention.size())
            attention_context = torch.bmm(F.softmax(attention,dim=1).transpose(1, 2), option_out)
            attention_option = torch.bmm(F.softmax(attention,dim=2), context_out)
#            print(attention_context.size())
#            print(attention_option.size())
            context_concat_out, context_concat_hid = self.rnn2(torch.cat((context_out, attention_context, context_out*attention_context, context_out-attention_context), -1))
            option_concat_out, option_concat_hid = self.rnn2(torch.cat((option_out, attention_option, option_out*attention_option, option_out-attention_option), -1))
#            print(context_concat_out.size())
#            print(option_concat_out.size())
            context_max = context_concat_out.max(dim=1)[0]
            option_max = option_concat_out.max(dim=1)[0]
#            print(context_max.size())
#            print(option_max.size())
            logit = self.similarity(torch.cat((context_max, option_max), -1))
#            print(logit.size())
            logits.append(logit.squeeze(dim=-1))
        logits = F.softmax(torch.stack(logits, 1), dim=1)
        return logits
