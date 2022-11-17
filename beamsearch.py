from transformers import GPT2Tokenizer
from my_gpt2 import GPT2LMHeadModel
import random
import torch
import copy


model_path = './models/gpt-2/'
data_path  = './data/'
tokenizer = GPT2Tokenizer.from_pretrained(model_path) 

DEVICE = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

def select_top_k(predictions, k=10):
    '''
    执行top-k选择，避免出现单词循环现象的出现
    @params prediction: GPT-2模型对下一个词的预测向量
    @params     k     : 默认 k=10
    '''
    predicted_index = predictions[0, -1, :].sort(descending=True)[0][:k]
    predicted_score = predictions[0, -1, :].sort(descending=True)[1][:k]

    return predicted_index,predicted_score

def beam_search(decoder, num_beams, max_len, input):
    """
    a beam search implementation about seq2seq with attention
    :param decoder:
    :param num_beams: number of beam, int
    :param max_len: max length of result
    :param input: input of decoder
    :return: list of index
    """
    # init 
    beams = [ [0,input] ]

    for i in range(max_len):
        results = []
        sorted_scores = []

        for beam in beams:
            indexed_tokens = beam[1]
            cur_score = beam[0]
            # print(indexed_tokens,cur_score)
            tokens_tensor = torch.tensor([indexed_tokens]).to(DEVICE)
            # print(tokens_tensor.shape)

            predictions = decoder(tokens_tensor)[0]
            # print(predictions.shape)
            score,index = select_top_k(predictions,num_beams)
            # tmp = sorted([(score, idx) for idx, score in enumerate(predictions[0, -1, :])],reverse=True)[:num_beams]
            # print(tmp)
            for i in range(num_beams):
                # print(index[i].item(),score[i].item())
                this_tokens = copy.deepcopy(indexed_tokens)
                this_tokens.append(index[i].item())
                results.append([this_tokens,cur_score+score[i].item()])
            sorted_scores = sorted([(score, idx) for idx, score in results],reverse=True)[:num_beams]
        
        # print(sorted_scores)
        beams = sorted_scores
    return beams[0][1]  # beams[0] has the highest score, beams[0][1] corresponding to the index


def do_test(text,length,model):
    '''
    模型根据前文内容预测后续句子
    @params text: 前文
    @params length: 后续句子的长度
    @params model: 使用的预训练模型(GPT-2)
    '''
    indexed_tokens = tokenizer.encode(text)
    model.eval()
    tokens = beam_search(model,num_beams=3,max_len=length,input=indexed_tokens)
    
    # print(tokenizer.decode(tokens))

def main():
    random.seed(42)
    ## load model
    model = GPT2LMHeadModel.from_pretrained(model_path+'pytorch_model.bin',config=model_path+'config.json')
    model.to(DEVICE)
    ## do a test
    do_test(text="My name is Mark. I am a student. His name is Jone, he is a",length=1,model=model)

        
if __name__ == "__main__":
    main()