# lm

# simple
group_dict = {
        'embedding' : ['encoder.weight'],
        'rnn1' : ['rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'rnn.bias_ih_l0', 'rnn.bias_hh_l0'],
        'rnn2' : ['rnn.weight_ih_l1', 'rnn.weight_hh_l1', 'rnn.bias_ih_l1', 'rnn.bias_hh_l1'],
        'decoder' : ['decoder.weight','decoder.bias']
        }
