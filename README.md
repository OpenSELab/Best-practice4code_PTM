# This is the data and code repository for manuscripts submitted to EMSE journal.

## Datasets
We first give the datasets under the datasets folder shown above, these datasets were processed well by previous researchers.

## Code Pre-trained Models
The code pre-trained models used in this paper are all come from HuggingFace, we give the links to download them:

CodeBERT-base: https://huggingface.co/microsoft/codebert-base

CodeT5-base: https://huggingface.co/Salesforce/codet5-base

PLBART-csnet: https://huggingface.co/uclanlp/plbart-csnet

CodeGPT: https://huggingface.co/microsoft/CodeGPT-small-py-adaptedGPT2

CodeGen-base: https://huggingface.co/Salesforce/codegen-350M-mono

We do not give the models shown above since it is too large and easy to obtain.


## Different Embedding Ways
The embedding method can be found in detail in the paper, and this step can be easily implemented. 

We provide code blocks to obtain embeddings by special tokens and average-pooling all codes tokens. 

These code chunks can be put into the pipeline where the pre-trained model is applied to downstream tasks (e.g., the pipeline provided by CodeXGLUE) to obtain code embeddings, provided that the model parameters are first frozen.


       
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, token_embeddings, attention_mask):
          #token_embeddings = model_output[0] #First element of model_output contains all token embeddings
          print("token_embeddings1-",np.shape(token_embeddings),token_embeddings) #1,7,384
          print("attention_mask1-",np.shape(attention_mask),attention_mask) #1,7,384
          token_embeddings = token_embeddings[:, 1:, :] #去除CLS标记
          attention_mask = attention_mask[:, 1:] #去除CLS标记对应的注意力
          print("token_embeddings2-",np.shape(token_embeddings),token_embeddings) #1,7,384
          print("attention_mask2-",np.shape(attention_mask),attention_mask) #1,7,384
          # 将每一行的最后一个 True 改为 False 这是对应SEP的注意力
          for row in attention_mask:
               last_true_index = (row == True).nonzero(as_tuple=True)[0][-1]
               row[last_true_index] = False
          print("attention_mask3-",np.shape(attention_mask),attention_mask) #1,7,384

          input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
          return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


        #x_commit = self.sentence_encoder(input_ids = input_ids, attention_mask=input_ids.ne(1))[0][:,0,:]


        #x_commit = self.sentence_encoder(input_ids = input_ids, attention_mask=input_ids.ne(1))[0][:,-1,:] #最后一个标记上的内容
#Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, token_embeddings, attention_mask):
          #token_embeddings = model_output[0] #First element of model_output contains all token embeddings
          print("token_embeddings1-",np.shape(token_embeddings),token_embeddings) #1,7,384
          print("attention_mask1-",np.shape(attention_mask),attention_mask) #1,7,384
          token_embeddings = token_embeddings[:,:-1,:]
          attention_mask = attention_mask[:,:-1]
          print("token_embeddings2-",np.shape(token_embeddings),token_embeddings) #1,7,384
          print("attention_mask2-",np.shape(attention_mask),attention_mask) #1,7,384
          input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
          print("input_mask_expanded -",np.shape(input_mask_expanded ),input_mask_expanded ) #1,7,384
          return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
       

        #######bos+eos########
        print(np.shape(hidden_states))
        print(hidden_states)
        #bos => eos 互换分别得到第一个和最后一个 这一行代码创建了一个名为 eos_mask 的布尔型张量，用于指示哪些位置包含了T5的结束标记（<eos>）。这是通过检查输入 source_ids 是否等于T5配置中的结束标记ID来实现的。
        eos_mask = source_ids.eq(self.sentence_encoder.config.eos_token_id)
        print(np.shape(eos_mask))
        print(eos_mask)
        if len(torch.unique(eos_mask.sum(1))) > 1:
           raise ValueError("All examples must have the same number of <eos> tokens.")

        #这一行代码根据 eos_mask 从隐藏状态中选择包含结束标记的位置，然后将这些位置的隐藏状态合并成一个向量。最后的结果是一个二维张量，每一行对应一个输入示例，每一行中的向量是该示例的编码表示。
        x_commit = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]
        print(np.shape(x_commit))
        print(x_commit)
        #######bos+eos########


        

## Acknowledgment
Thank the following works for providing the original repository to facilitate the collection of the data sets required for this paper. At the same time, we gave the address of the raw datasets depository.

Datasets Devign and BigCloneBench for code vulnerability detection and code clone detection
Devign: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection
and the raw dataset comes from the paper <b>Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks.</b>

Datasets CWE119 and CWE399: https://github.com/CGCL-codes/VulDeePecker
and these two datasets come from the paper <b>Vuldeepecker: A deep learning-based system for vulnerability detection.</b>

BigCloneBench: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench
and the raw dataset comes from the paper <b>Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree.</b>

Datasets openstack, qt, platform, gerrit, and go for just-in-time defect prediction: https://github.com/ZZR0/ISSTA21-JIT-DP
and the raw dataset comes from the paper <b>Deep just-in-time defect prediction: how far are we?</b>

