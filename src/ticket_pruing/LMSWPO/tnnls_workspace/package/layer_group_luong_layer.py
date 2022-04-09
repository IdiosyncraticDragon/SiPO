# Luong Net
# group_dict={
# 'all':['encoder.embeddings.make_embedding.emb_luts.0.weight','encoder.rnn.weight_ih_l0', 'encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_ih_l0', 'encoder.rnn.bias_hh_l0','encoder.rnn.weight_ih_l1', 'encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_ih_l1', 'encoder.rnn.bias_hh_l1','encoder.rnn.weight_ih_l2', 'encoder.rnn.weight_hh_l2', 'encoder.rnn.bias_ih_l2', 'encoder.rnn.bias_hh_l2', 'encoder.rnn.weight_ih_l3', 'encoder.rnn.weight_hh_l3', 'encoder.rnn.bias_ih_l3', 'encoder.rnn.bias_hh_l3', 'decoder.embeddings.make_embedding.emb_luts.0.weight', 'decoder.rnn.layers.0.weight_ih', 'decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_ih', 'decoder.rnn.layers.0.bias_hh', 'decoder.rnn.layers.1.weight_ih', 'decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_ih', 'decoder.rnn.layers.1.bias_hh', 'decoder.rnn.layers.2.weight_ih', 'decoder.rnn.layers.2.weight_hh', 'decoder.rnn.layers.2.bias_ih', 'decoder.rnn.layers.2.bias_hh', 'decoder.rnn.layers.3.weight_ih', 'decoder.rnn.layers.3.weight_hh', 'decoder.rnn.layers.3.bias_ih', 'decoder.rnn.layers.3.bias_hh', 'decoder.attn.linear_in.weight', 'decoder.attn.linear_out.weight', 'generator.0.weight', 'generator.0.bias']
# }
'''
time-wise
group_dict={
'embedding':['encoder.embeddings.make_embedding.emb_luts.0.weight','decoder.embeddings.make_embedding.emb_luts.0.weight','generator.0.weight', 'generator.0.bias'],\
'ih':['encoder.rnn.weight_ih_l0','encoder.rnn.weight_ih_l1','encoder.rnn.weight_ih_l2','encoder.rnn.weight_ih_l3','decoder.rnn.layers.0.weight_ih','decoder.rnn.layers.1.weight_ih','decoder.rnn.layers.2.weight_ih','decoder.rnn.layers.3.weight_ih','encoder.rnn.bias_ih_l0','encoder.rnn.bias_ih_l1','encoder.rnn.bias_ih_l2', 'encoder.rnn.bias_ih_l3', 'decoder.rnn.layers.0.bias_ih', 'decoder.rnn.layers.1.bias_ih', 'decoder.rnn.layers.2.bias_ih', 'decoder.rnn.layers.3.bias_ih'],\
'hh':['encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_hh_l0','encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_hh_l1', 'encoder.rnn.weight_hh_l2',  'encoder.rnn.bias_hh_l2', 'encoder.rnn.weight_hh_l3', 'encoder.rnn.bias_hh_l3', 'decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_hh', 'decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_hh', 'decoder.rnn.layers.2.weight_hh', 'decoder.rnn.layers.2.bias_hh', 'decoder.rnn.layers.3.weight_hh', 'decoder.rnn.layers.3.bias_hh'],\
'attn':['decoder.attn.linear_in.weight', 'decoder.attn.linear_out.weight']
}
'''

#layer-wise
group_dict={
'encoder_embedding':['encoder.embeddings.make_embedding.emb_luts.0.weight'],\
'encoder_rnn0':['encoder.rnn.weight_ih_l0', 'encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_ih_l0', 'encoder.rnn.bias_hh_l0'],\
'encoder_rnn1':['encoder.rnn.weight_ih_l1', 'encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_ih_l1', 'encoder.rnn.bias_hh_l1'],\
'encoder_rnn2':['encoder.rnn.weight_ih_l2', 'encoder.rnn.weight_hh_l2', 'encoder.rnn.bias_ih_l2', 'encoder.rnn.bias_hh_l2'],\
'encoder_rnn3':['encoder.rnn.weight_ih_l3', 'encoder.rnn.weight_hh_l3', 'encoder.rnn.bias_ih_l3', 'encoder.rnn.bias_hh_l3'],\
'decoder_embedding':['decoder.embeddings.make_embedding.emb_luts.0.weight'],\
'decoder_rnn0':['decoder.rnn.layers.0.weight_ih', 'decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_ih', 'decoder.rnn.layers.0.bias_hh'],\
'decoder_rnn1':['decoder.rnn.layers.1.weight_ih', 'decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_ih', 'decoder.rnn.layers.1.bias_hh'],\
'decoder_rnn2':['decoder.rnn.layers.2.weight_ih', 'decoder.rnn.layers.2.weight_hh', 'decoder.rnn.layers.2.bias_ih', 'decoder.rnn.layers.2.bias_hh'],\
'decoder_rnn3':['decoder.rnn.layers.3.weight_ih', 'decoder.rnn.layers.3.weight_hh', 'decoder.rnn.layers.3.bias_ih', 'decoder.rnn.layers.3.bias_hh'],\
'attn':['decoder.attn.linear_in.weight', 'decoder.attn.linear_out.weight'],\
'pred':['generator.0.weight', 'generator.0.bias']
}


'''
group_dict1={
'encoder_rnn01':['encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_hh_l0'],\
'encoder_rnn11':['encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_hh_l1'],\
'encoder_rnn21':['encoder.rnn.weight_hh_l2', 'encoder.rnn.bias_hh_l2'],\
'encoder_rnn31':[ 'encoder.rnn.weight_hh_l3', 'encoder.rnn.bias_hh_l3'],\
'decoder_rnn01':['decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_hh'],\
'decoder_rnn11':['decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_hh'],\
'decoder_rnn21':['decoder.rnn.layers.2.weight_hh', 'decoder.rnn.layers.2.bias_hh'],\
'decoder_rnn31':['decoder.rnn.layers.3.weight_hh', 'decoder.rnn.layers.3.bias_hh'],\
}
'''
'''
time-wise+layer-wise
group_dict={
'embedding':['encoder.embeddings.make_embedding.emb_luts.0.weight','decoder.embeddings.make_embedding.emb_luts.0.weight','generator.0.weight', 'generator.0.bias'],\
'encoder_rnn0':['encoder.rnn.weight_ih_l0', 'encoder.rnn.bias_ih_l0'],\
'encoder_rnn01':['encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_hh_l0'],\
'encoder_rnn1':['encoder.rnn.weight_ih_l1', 'encoder.rnn.bias_ih_l1'],\
'encoder_rnn11':['encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_hh_l1'],\
'encoder_rnn2':['encoder.rnn.weight_ih_l2', 'encoder.rnn.bias_ih_l2'],\
'encoder_rnn21':['encoder.rnn.weight_hh_l2', 'encoder.rnn.bias_hh_l2'],\
'encoder_rnn3':['encoder.rnn.weight_ih_l3', 'encoder.rnn.bias_ih_l3'],\
'encoder_rnn31':[ 'encoder.rnn.weight_hh_l3', 'encoder.rnn.bias_hh_l3'],\
'decoder_rnn0':['decoder.rnn.layers.0.weight_ih', 'decoder.rnn.layers.0.bias_ih'],\
'decoder_rnn01':['decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_hh'],\
'decoder_rnn1':['decoder.rnn.layers.1.weight_ih', 'decoder.rnn.layers.1.bias_ih'],\
'decoder_rnn11':['decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_hh'],\
'decoder_rnn2':['decoder.rnn.layers.2.weight_ih', 'decoder.rnn.layers.2.bias_ih'],\
'decoder_rnn21':['decoder.rnn.layers.2.weight_hh', 'decoder.rnn.layers.2.bias_hh'],\
'decoder_rnn3':['decoder.rnn.layers.3.weight_ih', 'decoder.rnn.layers.3.bias_ih'],\
'decoder_rnn31':['decoder.rnn.layers.3.weight_hh', 'decoder.rnn.layers.3.bias_hh'],\
'attn':['decoder.attn.linear_in.weight', 'decoder.attn.linear_out.weight']
}
'''

#{'encoder.embeddings.make_embedding.emb_luts.0.weight': 0, 'encoder.rnn.weight_ih_l0': 1, 'encoder.rnn.weight_hh_l0': 2, 'encoder.rnn.bias_ih_l0': 3, 'encoder.rnn.bias_hh_l0': 4, 'encoder.rnn.weight_ih_l1': 5, 'encoder.rnn.weight_hh_l1': 6, 'encoder.rnn.bias_ih_l1': 7, 'encoder.rnn.bias_hh_l1': 8, 'encoder.rnn.weight_ih_l2': 9, 'encoder.rnn.weight_hh_l2': 10, 'encoder.rnn.bias_ih_l2': 11, 'encoder.rnn.bias_hh_l2': 12, 'encoder.rnn.weight_ih_l3': 13, 'encoder.rnn.weight_hh_l3': 14, 'encoder.rnn.bias_ih_l3': 15, 'encoder.rnn.bias_hh_l3': 16, 'decoder.embeddings.make_embedding.emb_luts.0.weight': 17, 'decoder.rnn.layers.0.weight_ih': 18, 'decoder.rnn.layers.0.weight_hh': 19, 'decoder.rnn.layers.0.bias_ih': 20, 'decoder.rnn.layers.0.bias_hh': 21, 'decoder.rnn.layers.1.weight_ih': 22, 'decoder.rnn.layers.1.weight_hh': 23, 'decoder.rnn.layers.1.bias_ih': 24, 'decoder.rnn.layers.1.bias_hh': 25, 'decoder.rnn.layers.2.weight_ih': 26, 'decoder.rnn.layers.2.weight_hh': 27, 'decoder.rnn.layers.2.bias_ih': 28, 'decoder.rnn.layers.2.bias_hh': 29, 'decoder.rnn.layers.3.weight_ih': 30, 'decoder.rnn.layers.3.weight_hh': 31, 'decoder.rnn.layers.3.bias_ih': 32, 'decoder.rnn.layers.3.bias_hh': 33, 'decoder.attn.linear_in.weight': 34, 'decoder.attn.linear_out.weight': 35, 'generator.0.weight': 36, 'generator.0.bias': 37}

## Bahandanu network
# RNN Search

#['encoder.embeddings.make_embedding.emb_luts.0.weight', 'encoder.rnn.weight_ih_l0', 'encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_ih_l0', 'encoder.rnn.bias_hh_l0', 'encoder.rnn.weight_ih_l0_reverse', 'encoder.rnn.weight_hh_l0_reverse', 'encoder.rnn.bias_ih_l0_reverse', 'encoder.rnn.bias_hh_l0_reverse', 'encoder.rnn.weight_ih_l1', 'encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_ih_l1', 'encoder.rnn.bias_hh_l1', 'encoder.rnn.weight_ih_l1_reverse', 'encoder.rnn.weight_hh_l1_reverse', 'encoder.rnn.bias_ih_l1_reverse', 'encoder.rnn.bias_hh_l1_reverse', 'decoder.embeddings.make_embedding.emb_luts.0.weight', 'decoder.rnn.layers.0.weight_ih', 'decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_ih', 'decoder.rnn.layers.0.bias_hh', 'decoder.rnn.layers.1.weight_ih', 'decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_ih', 'decoder.rnn.layers.1.bias_hh', 'decoder.attn.linear_context.weight', 'decoder.attn.linear_query.weight', 'decoder.attn.linear_query.bias', 'decoder.attn.v.weight', 'decoder.attn.linear_out.weight', 'decoder.attn.linear_out.bias', 'generator.0.weight', 'generator.0.bias']
'''
group_dict={
'encoder_embedding':['encoder.embeddings.make_embedding.emb_luts.0.weight'],\
'decoder_embedding':['decoder.embeddings.make_embedding.emb_luts.0.weight'],\
'encoder_rnn0ih':['encoder.rnn.weight_ih_l0', 'encoder.rnn.bias_ih_l0'],\
'encoder_rnn0hh':['encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_hh_l0'],\
'encoder_rnn0ih_reverse':['encoder.rnn.weight_ih_l0_reverse', 'encoder.rnn.bias_ih_l0_reverse'],\
'encoder_rnn0hh_reverse':['encoder.rnn.weight_hh_l0_reverse', 'encoder.rnn.bias_hh_l0_reverse'],\
'encoder_rnn1ih':['encoder.rnn.weight_ih_l1', 'encoder.rnn.bias_ih_l1'],\
'encoder_rnn1hh':['encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_hh_l1'],\
'encoder_rnn1ih_reverse':['encoder.rnn.weight_ih_l1_reverse', 'encoder.rnn.bias_ih_l1_reverse'],\
'encoder_rnn1hh_reverse':['encoder.rnn.weight_hh_l1_reverse', 'encoder.rnn.bias_hh_l1_reverse'],\
'decoder_rnn0ih':['decoder.rnn.layers.0.weight_ih', 'decoder.rnn.layers.0.bias_ih'],\
'decoder_rnn0hh':['decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_hh'],\
'decoder_rnn0ih_reverse':['decoder.rnn.layers.0.weight_ih_reverse', 'decoder.rnn.layers.0.bias_ih_reverse'],\
'decoder_rnn0hh_reverse':['decoder.rnn.layers.0.weight_hh_reverse', 'decoder.rnn.layers.0.bias_hh_reverse'],\
'decoder_rnn1ih':['decoder.rnn.layers.1.weight_ih', 'decoder.rnn.layers.1.bias_ih'],\
'decoder_rnn1hh':['decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_hh'],\
'decoder_rnn1ih_reverse':['decoder.rnn.layers.1.weight_ih_reverse', 'decoder.rnn.layers.1.bias_ih_reverse'],\
'decoder_rnn1hh_reverse':['decoder.rnn.layers.1.weight_hh_reverse', 'decoder.rnn.layers.1.bias_hh_reverse'],\
'attn':['decoder.attn.linear_context.weight', 'decoder.attn.linear_query.weight', 'decoder.attn.linear_query.bias', 'decoder.attn.v.weight', 'decoder.attn.linear_out.bias', 'decoder.attn.linear_in.weight', 'decoder.attn.linear_out.weight'],\
'pred':['generator.0.weight', 'generator.0.bias']
}
'''
'''
group_dict={
'encoder_embedding':['encoder.embeddings.make_embedding.emb_luts.0.weight','decoder.embeddings.make_embedding.emb_luts.0.weight','generator.0.weight', 'generator.0.bias'],\
#'encoder_embedding':['encoder.embeddings.make_embedding.emb_luts.0.weight'],\
'encoder_rnn0':['encoder.rnn.weight_ih_l0', 'encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_ih_l0', 'encoder.rnn.bias_hh_l0'],\
'encoder_rnn0_reverse':['encoder.rnn.weight_ih_l0_reverse', 'encoder.rnn.weight_hh_l0_reverse', 'encoder.rnn.bias_ih_l0_reverse', 'encoder.rnn.bias_hh_l0_reverse'],\
'encoder_rnn1':['encoder.rnn.weight_ih_l1', 'encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_ih_l1', 'encoder.rnn.bias_hh_l1'],\
'encoder_rnn1_reverse':['encoder.rnn.weight_ih_l1_reverse', 'encoder.rnn.weight_hh_l1_reverse', 'encoder.rnn.bias_ih_l1_reverse', 'encoder.rnn.bias_hh_l1_reverse'],\
#'decoder_embedding':['decoder.embeddings.make_embedding.emb_luts.0.weight'],\
'decoder_rnn0':['decoder.rnn.layers.0.weight_ih', 'decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_ih', 'decoder.rnn.layers.0.bias_hh'],\
'decoder_rnn0_reverse':['decoder.rnn.layers.0.weight_ih_reverse', 'decoder.rnn.layers.0.weight_hh_reverse', 'decoder.rnn.layers.0.bias_ih_reverse', 'decoder.rnn.layers.0.bias_hh_reverse'],\
'decoder_rnn1':['decoder.rnn.layers.1.weight_ih', 'decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_ih', 'decoder.rnn.layers.1.bias_hh'],\
'decoder_rnn1_reverse':['decoder.rnn.layers.1.weight_ih_reverse', 'decoder.rnn.layers.1.weight_hh_reverse', 'decoder.rnn.layers.1.bias_ih_reverse', 'decoder.rnn.layers.1.bias_hh_reverse'],\
'attn':['decoder.attn.linear_context.weight', 'decoder.attn.linear_query.weight', 'decoder.attn.linear_query.bias', 'decoder.attn.v.weight', 'decoder.attn.linear_out.bias', 'decoder.attn.linear_in.weight', 'decoder.attn.linear_out.weight'],\
#'pred':['generator.0.weight', 'generator.0.bias']
}
'''
'''
group_dict1={
#'encoder_embedding':['encoder.embeddings.make_embedding.emb_luts.0.weight'],\
#'decoder_embedding':['decoder.embeddings.make_embedding.emb_luts.0.weight'],\
#'encoder_rnn0ih':['encoder.rnn.weight_ih_l0', 'encoder.rnn.bias_ih_l0'],\
'encoder_rnn0hh':['encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_hh_l0'],\
#'encoder_rnn0ih_reverse':['encoder.rnn.weight_ih_l0_reverse', 'encoder.rnn.bias_ih_l0_reverse'],\
'encoder_rnn0hh_reverse':['encoder.rnn.weight_hh_l0_reverse', 'encoder.rnn.bias_hh_l0_reverse'],\
#'encoder_rnn1ih':['encoder.rnn.weight_ih_l1', 'encoder.rnn.bias_ih_l1'],\
'encoder_rnn1hh':['encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_hh_l1'],\
#'encoder_rnn1ih_reverse':['encoder.rnn.weight_ih_l1_reverse', 'encoder.rnn.bias_ih_l1_reverse'],\
'encoder_rnn1hh_reverse':['encoder.rnn.weight_hh_l1_reverse', 'encoder.rnn.bias_hh_l1_reverse'],\
#'decoder_rnn0ih':['decoder.rnn.layers.0.weight_ih', 'decoder.rnn.layers.0.bias_ih'],\
'decoder_rnn0hh':['decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_hh'],\
#'decoder_rnn0ih_reverse':['decoder.rnn.layers.0.weight_ih_reverse', 'decoder.rnn.layers.0.bias_ih_reverse'],\
'decoder_rnn0hh_reverse':['decoder.rnn.layers.0.weight_hh_reverse', 'decoder.rnn.layers.0.bias_hh_reverse'],\
#'decoder_rnn1ih':['decoder.rnn.layers.1.weight_ih', 'decoder.rnn.layers.1.bias_ih'],\
'decoder_rnn1hh':['decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_hh'],\
#'decoder_rnn1ih_reverse':['decoder.rnn.layers.1.weight_ih_reverse', 'decoder.rnn.layers.1.bias_ih_reverse'],\
'decoder_rnn1hh_reverse':['decoder.rnn.layers.1.weight_hh_reverse', 'decoder.rnn.layers.1.bias_hh_reverse'],\
#'attn':['decoder.attn.linear_context.weight', 'decoder.attn.linear_query.weight', 'decoder.attn.linear_query.bias', 'decoder.attn.v.weight', 'decoder.attn.linear_out.bias', 'decoder.attn.linear_in.weight', 'decoder.attn.linear_out.weight'],\
#'pred':['generator.0.weight', 'generator.0.bias']
}
group_dict2={
'encoder_embedding':['encoder.embeddings.make_embedding.emb_luts.0.weight'],\
'decoder_embedding':['decoder.embeddings.make_embedding.emb_luts.0.weight'],\
'encoder_rnn0ih':['encoder.rnn.weight_ih_l0', 'encoder.rnn.bias_ih_l0'],\
#'encoder_rnn0hh':['encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_hh_l0'],\
'encoder_rnn0ih_reverse':['encoder.rnn.weight_ih_l0_reverse', 'encoder.rnn.bias_ih_l0_reverse'],\
#'encoder_rnn0hh_reverse':['encoder.rnn.weight_hh_l0_reverse', 'encoder.rnn.bias_hh_l0_reverse'],\
'encoder_rnn1ih':['encoder.rnn.weight_ih_l1', 'encoder.rnn.bias_ih_l1'],\
#'encoder_rnn1hh':['encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_hh_l1'],\
'encoder_rnn1ih_reverse':['encoder.rnn.weight_ih_l1_reverse', 'encoder.rnn.bias_ih_l1_reverse'],\
#'encoder_rnn1hh_reverse':['encoder.rnn.weight_hh_l1_reverse', 'encoder.rnn.bias_hh_l1_reverse'],\
'decoder_rnn0ih':['decoder.rnn.layers.0.weight_ih', 'decoder.rnn.layers.0.bias_ih'],\
#'decoder_rnn0hh':['decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_hh'],\
'decoder_rnn0ih_reverse':['decoder.rnn.layers.0.weight_ih_reverse', 'decoder.rnn.layers.0.bias_ih_reverse'],\
#'decoder_rnn0hh_reverse':['decoder.rnn.layers.0.weight_hh_reverse', 'decoder.rnn.layers.0.bias_hh_reverse'],\
'decoder_rnn1ih':['decoder.rnn.layers.1.weight_ih', 'decoder.rnn.layers.1.bias_ih'],\
#'decoder_rnn1hh':['decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_hh'],\
'decoder_rnn1ih_reverse':['decoder.rnn.layers.1.weight_ih_reverse', 'decoder.rnn.layers.1.bias_ih_reverse'],\
#'decoder_rnn1hh_reverse':['decoder.rnn.layers.1.weight_hh_reverse', 'decoder.rnn.layers.1.bias_hh_reverse'],\
'attn':['decoder.attn.linear_context.weight', 'decoder.attn.linear_query.weight', 'decoder.attn.linear_query.bias', 'decoder.attn.v.weight', 'decoder.attn.linear_out.bias', 'decoder.attn.linear_in.weight', 'decoder.attn.linear_out.weight'],\
'pred':['generator.0.weight', 'generator.0.bias']
}
'''
'''
group_dict={
'all':['encoder.embeddings.make_embedding.emb_luts.0.weight','encoder.rnn.weight_ih_l0', 'encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_ih_l0', 'encoder.rnn.bias_hh_l0','encoder.rnn.weight_ih_l0_reverse', 'encoder.rnn.weight_hh_l0_reverse', 'encoder.rnn.bias_ih_l0_reverse', 'encoder.rnn.bias_hh_l0_reverse','encoder.rnn.weight_ih_l1', 'encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_ih_l1', 'encoder.rnn.bias_hh_l1','encoder.rnn.weight_ih_l1_reverse', 'encoder.rnn.weight_hh_l1_reverse', 'encoder.rnn.bias_ih_l1_reverse', 'encoder.rnn.bias_hh_l1_reverse','decoder.embeddings.make_embedding.emb_luts.0.weight','decoder.rnn.layers.0.weight_ih', 'decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_ih', 'decoder.rnn.layers.0.bias_hh','decoder.rnn.layers.0.weight_ih_reverse', 'decoder.rnn.layers.0.weight_hh_reverse', 'decoder.rnn.layers.0.bias_ih_reverse', 'decoder.rnn.layers.0.bias_hh_reverse','decoder.rnn.layers.1.weight_ih', 'decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_ih', 'decoder.rnn.layers.1.bias_hh','decoder.rnn.layers.1.weight_ih_reverse', 'decoder.rnn.layers.1.weight_hh_reverse', 'decoder.rnn.layers.1.bias_ih_reverse', 'decoder.rnn.layers.1.bias_hh_reverse','decoder.attn.linear_context.weight', 'decoder.attn.linear_query.weight', 'decoder.attn.linear_query.bias', 'decoder.attn.v.weight', 'decoder.attn.linear_out.bias', 'decoder.attn.linear_in.weight', 'decoder.attn.linear_out.weight','generator.0.weight', 'generator.0.bias']
}
'''
'''
# layer-wise
group_dict={
'encoder_embedding':['encoder.embeddings.make_embedding.emb_luts.0.weight'],\
'decoder_embedding':['decoder.embeddings.make_embedding.emb_luts.0.weight'],\
'encoder_rnn0':['encoder.rnn.weight_ih_l0', 'encoder.rnn.bias_ih_l0','encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_hh_l0','encoder.rnn.weight_ih_l0_reverse', 'encoder.rnn.bias_ih_l0_reverse','encoder.rnn.weight_hh_l0_reverse', 'encoder.rnn.bias_hh_l0_reverse'],\
'encoder_rnn1':['encoder.rnn.weight_ih_l1', 'encoder.rnn.bias_ih_l1','encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_hh_l1','encoder.rnn.weight_ih_l1_reverse', 'encoder.rnn.bias_ih_l1_reverse','encoder.rnn.weight_hh_l1_reverse', 'encoder.rnn.bias_hh_l1_reverse'],\
'decoder_rnn0':['decoder.rnn.layers.0.weight_ih', 'decoder.rnn.layers.0.bias_ih','decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_hh','decoder.rnn.layers.0.weight_ih_reverse', 'decoder.rnn.layers.0.bias_ih_reverse','decoder.rnn.layers.0.weight_hh_reverse', 'decoder.rnn.layers.0.bias_hh_reverse'],\
'decoder_rnn1':['decoder.rnn.layers.1.weight_ih', 'decoder.rnn.layers.1.bias_ih','decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_hh','decoder.rnn.layers.1.weight_ih_reverse', 'decoder.rnn.layers.1.bias_ih_reverse','decoder.rnn.layers.1.weight_hh_reverse', 'decoder.rnn.layers.1.bias_hh_reverse'],\
'attn':['decoder.attn.linear_context.weight', 'decoder.attn.linear_query.weight', 'decoder.attn.linear_query.bias', 'decoder.attn.v.weight', 'decoder.attn.linear_out.bias', 'decoder.attn.linear_in.weight', 'decoder.attn.linear_out.weight'],\
'pred':['generator.0.weight', 'generator.0.bias']
}
'''
'''
# time-wise
group_dict={
'encoder_embedding':['encoder.embeddings.make_embedding.emb_luts.0.weight'],\
'decoder_embedding':['decoder.embeddings.make_embedding.emb_luts.0.weight'],\
'hh':['encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_hh_l0','encoder.rnn.weight_hh_l0_reverse', 'encoder.rnn.bias_hh_l0_reverse','encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_hh_l1','encoder.rnn.weight_hh_l1_reverse', 'encoder.rnn.bias_hh_l1_reverse','decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_hh','decoder.rnn.layers.0.weight_hh_reverse', 'decoder.rnn.layers.0.bias_hh_reverse','decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_hh','decoder.rnn.layers.1.weight_hh_reverse', 'decoder.rnn.layers.1.bias_hh_reverse'],\
'ih':['encoder.rnn.weight_ih_l0', 'encoder.rnn.bias_ih_l0','encoder.rnn.weight_ih_l0_reverse', 'encoder.rnn.bias_ih_l0_reverse','encoder.rnn.weight_ih_l1','encoder.rnn.bias_ih_l1','encoder.rnn.weight_ih_l1_reverse', 'encoder.rnn.bias_ih_l1_reverse','decoder.rnn.layers.0.weight_ih', 'decoder.rnn.layers.0.bias_ih','decoder.rnn.layers.0.weight_ih_reverse', 'decoder.rnn.layers.0.bias_ih_reverse','decoder.rnn.layers.1.weight_ih', 'decoder.rnn.layers.1.bias_ih','decoder.rnn.layers.1.weight_ih_reverse', 'decoder.rnn.layers.1.bias_ih_reverse'],\
'attn':['decoder.attn.linear_context.weight', 'decoder.attn.linear_query.weight', 'decoder.attn.linear_query.bias', 'decoder.attn.v.weight', 'decoder.attn.linear_out.bias', 'decoder.attn.linear_in.weight', 'decoder.attn.linear_out.weight'],\
'pred':['generator.0.weight', 'generator.0.bias']
}
'''
'''
# time-wise + layer-wise
group_dict={
'encoder_embedding':['encoder.embeddings.make_embedding.emb_luts.0.weight'],\
'decoder_embedding':['decoder.embeddings.make_embedding.emb_luts.0.weight'],\
'encoder_rnn0ih':['encoder.rnn.weight_ih_l0', 'encoder.rnn.bias_ih_l0','encoder.rnn.weight_ih_l0_reverse', 'encoder.rnn.bias_ih_l0_reverse'],\
'encoder_rnn0hh':['encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_hh_l0','encoder.rnn.weight_hh_l0_reverse', 'encoder.rnn.bias_hh_l0_reverse'],\
'encoder_rnn1ih':['encoder.rnn.weight_ih_l1', 'encoder.rnn.bias_ih_l1','encoder.rnn.weight_ih_l1_reverse', 'encoder.rnn.bias_ih_l1_reverse'],\
'encoder_rnn1hh':['encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_hh_l1','encoder.rnn.weight_hh_l1_reverse', 'encoder.rnn.bias_hh_l1_reverse'],\
'decoder_rnn0ih':['decoder.rnn.layers.0.weight_ih', 'decoder.rnn.layers.0.bias_ih','decoder.rnn.layers.0.weight_ih_reverse', 'decoder.rnn.layers.0.bias_ih_reverse'],\
'decoder_rnn0hh':['decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_hh','decoder.rnn.layers.0.weight_hh_reverse', 'decoder.rnn.layers.0.bias_hh_reverse'],\
'decoder_rnn1ih':['decoder.rnn.layers.1.weight_ih', 'decoder.rnn.layers.1.bias_ih','decoder.rnn.layers.1.weight_ih_reverse', 'decoder.rnn.layers.1.bias_ih_reverse'],\
'decoder_rnn1hh':['decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_hh','decoder.rnn.layers.1.weight_hh_reverse', 'decoder.rnn.layers.1.bias_hh_reverse'],\
'attn':['decoder.attn.linear_context.weight', 'decoder.attn.linear_query.weight', 'decoder.attn.linear_query.bias', 'decoder.attn.v.weight', 'decoder.attn.linear_out.bias', 'decoder.attn.linear_in.weight', 'decoder.attn.linear_out.weight'],\
'pred':['generator.0.weight', 'generator.0.bias']
}
'''
'''
# bidirectional + layer-wise
group_dict={
'encoder_embedding':['encoder.embeddings.make_embedding.emb_luts.0.weight'],\
'decoder_embedding':['decoder.embeddings.make_embedding.emb_luts.0.weight'],\
'encoder_rnn0':['encoder.rnn.weight_ih_l0', 'encoder.rnn.bias_ih_l0','encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_hh_l0'],\
'encoder_rnn0_reverse':['encoder.rnn.weight_ih_l0_reverse', 'encoder.rnn.bias_ih_l0_reverse','encoder.rnn.weight_hh_l0_reverse', 'encoder.rnn.bias_hh_l0_reverse'],\
'encoder_rnn1':['encoder.rnn.weight_ih_l1', 'encoder.rnn.bias_ih_l1','encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_hh_l1'],\
'encoder_rnn1_reverse':['encoder.rnn.weight_ih_l1_reverse', 'encoder.rnn.bias_ih_l1_reverse','encoder.rnn.weight_hh_l1_reverse', 'encoder.rnn.bias_hh_l1_reverse'],\
'decoder_rnn0':['decoder.rnn.layers.0.weight_ih', 'decoder.rnn.layers.0.bias_ih','decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_hh'],\
'decoder_rnn0_reverse':['decoder.rnn.layers.0.weight_ih_reverse', 'decoder.rnn.layers.0.bias_ih_reverse','decoder.rnn.layers.0.weight_hh_reverse', 'decoder.rnn.layers.0.bias_hh_reverse'],\
'decoder_rnn1':['decoder.rnn.layers.1.weight_ih', 'decoder.rnn.layers.1.bias_ih','decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_hh'],\
'decoder_rnn1_reverse':['decoder.rnn.layers.1.weight_ih_reverse', 'decoder.rnn.layers.1.bias_ih_reverse','decoder.rnn.layers.1.weight_hh_reverse', 'decoder.rnn.layers.1.bias_hh_reverse'],\
'attn':['decoder.attn.linear_context.weight', 'decoder.attn.linear_query.weight', 'decoder.attn.linear_query.bias', 'decoder.attn.v.weight', 'decoder.attn.linear_out.bias', 'decoder.attn.linear_in.weight', 'decoder.attn.linear_out.weight'],\
'pred':['generator.0.weight', 'generator.0.bias']
}
'''
'''
# bidirectional + time-wise
group_dict = {
'encoder_embedding':['encoder.embeddings.make_embedding.emb_luts.0.weight'],\
'decoder_embedding':['decoder.embeddings.make_embedding.emb_luts.0.weight'],\
'hh':['encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_hh_l0','encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_hh_l1','decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_hh','decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_hh'],\
'hh_reverse':['encoder.rnn.weight_hh_l0_reverse', 'encoder.rnn.bias_hh_l0_reverse','encoder.rnn.weight_hh_l1_reverse', 'encoder.rnn.bias_hh_l1_reverse','decoder.rnn.layers.0.weight_hh_reverse', 'decoder.rnn.layers.0.bias_hh_reverse','decoder.rnn.layers.1.weight_hh_reverse', 'decoder.rnn.layers.1.bias_hh_reverse'],\
'ih':['encoder.rnn.weight_ih_l0', 'encoder.rnn.bias_ih_l0','encoder.rnn.weight_ih_l1', 'encoder.rnn.bias_ih_l1','decoder.rnn.layers.0.weight_ih', 'decoder.rnn.layers.0.bias_ih','decoder.rnn.layers.1.weight_ih', 'decoder.rnn.layers.1.bias_ih'],\
'ih_reverse':['encoder.rnn.weight_ih_l0_reverse', 'encoder.rnn.bias_ih_l0_reverse','encoder.rnn.weight_ih_l1_reverse', 'encoder.rnn.bias_ih_l1_reverse','decoder.rnn.layers.0.weight_ih_reverse', 'decoder.rnn.layers.0.bias_ih_reverse','decoder.rnn.layers.1.weight_ih_reverse', 'decoder.rnn.layers.1.bias_ih_reverse'],\
'attn':['decoder.attn.linear_context.weight', 'decoder.attn.linear_query.weight', 'decoder.attn.linear_query.bias', 'decoder.attn.v.weight', 'decoder.attn.linear_out.bias', 'decoder.attn.linear_in.weight', 'decoder.attn.linear_out.weight'],\
'pred':['generator.0.weight', 'generator.0.bias']
}
'''
'''
# bidirectional + layer-wise +  time-wise
group_dict={
'encoder_embedding':['encoder.embeddings.make_embedding.emb_luts.0.weight'],\
'decoder_embedding':['decoder.embeddings.make_embedding.emb_luts.0.weight'],\
'encoder_rnn0ih':['encoder.rnn.weight_ih_l0', 'encoder.rnn.bias_ih_l0'],\
'encoder_rnn0hh':['encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_hh_l0'],\
'encoder_rnn0ih_reverse':['encoder.rnn.weight_ih_l0_reverse', 'encoder.rnn.bias_ih_l0_reverse'],\
'encoder_rnn0hh_reverse':['encoder.rnn.weight_hh_l0_reverse', 'encoder.rnn.bias_hh_l0_reverse'],\
'encoder_rnn1ih':['encoder.rnn.weight_ih_l1', 'encoder.rnn.bias_ih_l1'],\
'encoder_rnn1hh':['encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_hh_l1'],\
'encoder_rnn1ih_reverse':['encoder.rnn.weight_ih_l1_reverse', 'encoder.rnn.bias_ih_l1_reverse'],\
'encoder_rnn1hh_reverse':['encoder.rnn.weight_hh_l1_reverse', 'encoder.rnn.bias_hh_l1_reverse'],\
'decoder_rnn0ih':['decoder.rnn.layers.0.weight_ih', 'decoder.rnn.layers.0.bias_ih'],\
'decoder_rnn0hh':['decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_hh'],\
'decoder_rnn0ih_reverse':['decoder.rnn.layers.0.weight_ih_reverse', 'decoder.rnn.layers.0.bias_ih_reverse'],\
'decoder_rnn0hh_reverse':['decoder.rnn.layers.0.weight_hh_reverse', 'decoder.rnn.layers.0.bias_hh_reverse'],\
'decoder_rnn1ih':['decoder.rnn.layers.1.weight_ih', 'decoder.rnn.layers.1.bias_ih'],\
'decoder_rnn1hh':['decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_hh'],\
'decoder_rnn1ih_reverse':['decoder.rnn.layers.1.weight_ih_reverse', 'decoder.rnn.layers.1.bias_ih_reverse'],\
'decoder_rnn1hh_reverse':['decoder.rnn.layers.1.weight_hh_reverse', 'decoder.rnn.layers.1.bias_hh_reverse'],\
'attn':['decoder.attn.linear_context.weight', 'decoder.attn.linear_query.weight', 'decoder.attn.linear_query.bias', 'decoder.attn.v.weight', 'decoder.attn.linear_out.bias', 'decoder.attn.linear_in.weight', 'decoder.attn.linear_out.weight'],\
'pred':['generator.0.weight', 'generator.0.bias']
}
'''
'''
# embedding combined with prediction + layer-wise
group_dict={
'embedding':['encoder.embeddings.make_embedding.emb_luts.0.weight','decoder.embeddings.make_embedding.emb_luts.0.weight','generator.0.weight', 'generator.0.bias'],\
'encoder_rnn0':['encoder.rnn.weight_ih_l0', 'encoder.rnn.bias_ih_l0','encoder.rnn.weight_hh_l0', 'encoder.rnn.bias_hh_l0','encoder.rnn.weight_ih_l0_reverse', 'encoder.rnn.bias_ih_l0_reverse','encoder.rnn.weight_hh_l0_reverse', 'encoder.rnn.bias_hh_l0_reverse'],\
'encoder_rnn1':['encoder.rnn.weight_ih_l1', 'encoder.rnn.bias_ih_l1','encoder.rnn.weight_hh_l1', 'encoder.rnn.bias_hh_l1','encoder.rnn.weight_ih_l1_reverse', 'encoder.rnn.bias_ih_l1_reverse','encoder.rnn.weight_hh_l1_reverse', 'encoder.rnn.bias_hh_l1_reverse'],\
'decoder_rnn0':['decoder.rnn.layers.0.weight_ih', 'decoder.rnn.layers.0.bias_ih','decoder.rnn.layers.0.weight_hh', 'decoder.rnn.layers.0.bias_hh','decoder.rnn.layers.0.weight_ih_reverse', 'decoder.rnn.layers.0.bias_ih_reverse','decoder.rnn.layers.0.weight_hh_reverse', 'decoder.rnn.layers.0.bias_hh_reverse'],\
'decoder_rnn1':['decoder.rnn.layers.1.weight_ih', 'decoder.rnn.layers.1.bias_ih','decoder.rnn.layers.1.weight_hh', 'decoder.rnn.layers.1.bias_hh','decoder.rnn.layers.1.weight_ih_reverse', 'decoder.rnn.layers.1.bias_ih_reverse','decoder.rnn.layers.1.weight_hh_reverse', 'decoder.rnn.layers.1.bias_hh_reverse'],\
'attn':['decoder.attn.linear_context.weight', 'decoder.attn.linear_query.weight', 'decoder.attn.linear_query.bias', 'decoder.attn.v.weight', 'decoder.attn.linear_out.bias', 'decoder.attn.linear_in.weight', 'decoder.attn.linear_out.weight']
}
'''
