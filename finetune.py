from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import trange

from transformers import GPT2Tokenizer
from my_gpt2 import GPT2LMHeadModel
import random
import torch


model_path = './models/gpt2-large/'
data_path  = './data/'
tokenizer = GPT2Tokenizer.from_pretrained(model_path) 
 
def select_top_k(predictions, k=10):
    '''
    执行top-k选择，避免出现单词循环现象的出现
    @params prediction: GPT-2模型对下一个词的预测向量
    @params     k     : 默认 k=10
    '''
    predicted_index = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_index


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping = False, src_length=0):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.src_length = src_length
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (len(hyp) - self.src_length) ** self.length_penalty
        # score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty


def preprocess_data(path):
    '''
    数据预处理
    @params path: 数据的路径
    return : List[int] 分成固定长度的"token索引片段"
    '''
    with open(path,'r') as f:
        dataset = f.read()

    # print(len(dataset))

    indexed_text = tokenizer.encode(dataset)
    del(dataset)

    dataset_cut = []
    for i in range(len(indexed_text)//512):
        # 将文本分段成 长度为512的片段
        dataset_cut.append(indexed_text[i*512:(i+1)*512])
    del(indexed_text)
    return dataset_cut

def do_train(epoch:int,optimizer:torch.optim.Optimizer,model,train_loader:DataLoader):
    '''
    进行模型训练
    @params epoch: 训练的循环次数
    @params optimizer: 参数优化器
    @params model: 使用的预训练模型(GPT-2)
    @params train_loader: 训练数据集
    '''
    model.train()
    for epoch in trange(epoch, desc="Epoch"):
        total_loss = 0
        for batch_idx ,(data,target) in enumerate(train_loader):
            optimizer.zero_grad()
            # print(batch_idx ,(data,target))
            loss = model(data,labels=target).loss
            total_loss += loss
            loss.backward()
            optimizer.step()
        print('average loss:',total_loss/len(train_loader))

def do_test(text,length,model):
    '''
    模型根据前文内容预测后续句子
    @params text: 前文
    @params length: 后续句子的长度
    @params model: 使用的预训练模型(GPT-2)
    '''
    indexed_tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([indexed_tokens])
    
    model.eval()
    total_predicted_text = text
    
    # 使训练后的模型进行 若干 次预测
    for _ in range(length):
        tokens_tensor = tokens_tensor.to('cuda')
    
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]
    
        predicted_index = select_top_k(predictions, k=1)
    
        total_predicted_text += tokenizer.decode(predicted_index)
        if '<|endoftext|>' in total_predicted_text:
            # 如果出现文本结束标志，就结束文本生成
            break
    
        indexed_tokens += [predicted_index]
    
        if len(indexed_tokens) > 1023:
            # 模型最长输入长度为1024，如果长度过长则截断
            indexed_tokens = indexed_tokens[-1023:]
    
        tokens_tensor = torch.tensor([indexed_tokens])
    
    print(total_predicted_text)

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## load data
    dataset_cut = preprocess_data(data_path+'romeo_and_juliet.txt')
    dataset_tensor = torch.tensor(dataset_cut).to(DEVICE)

    train_set = TensorDataset(dataset_tensor,dataset_tensor) # 标签与样本数据相同
    train_loader = DataLoader(dataset = train_set,batch_size = 2, shuffle=False)

    ## load model
    model = GPT2LMHeadModel.from_pretrained(model_path+'pytorch_model.bin',config=model_path+'config.json')
    model.to(DEVICE)

    epoch = 1
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)

    ## model training
    # do_train(epoch,optimizer,model,train_loader)
    random.seed(42)

    ## do a test
    do_test(text="|Name : David , Age : 18| Name : Tom , Age : 29| How old is David? A: 18. How old is Tom? A : ",length=2,model=model)
    

        
if __name__ == "__main__":
    main()