import torch
import pandas as pd
import nltk
import lmdb
from transformers import LxmertTokenizer, LxmertConfig, LxmertModel, LxmertForPreTraining, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import pickle

def flatten(tensor):
    nsamples, nx, ny = tensor.shape
    new_tensor = tensor.clone().reshape((nsamples,nx*ny))
    return new_tensor

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,dataset_path,image_path, lxmert_tokenizer, vsr = None, vsr_image_path = '../lxmert/data/vsr-images', max_length = 77):
        self.lxmert_tokenizer = lxmert_tokenizer
        self.max_length = max_length
        if(vsr):#assess only for the vsr dataset
            self.dataset = self.read_vsr_dataset(vsr)
            img_env = lmdb.open(
                vsr_image_path, readonly=True, create=False, readahead=not False
            )
            self.img_txn = img_env.begin(buffers=True)
        else:#assess for the SNLI-VE dataset
            self.dataset = self.read_dataset(dataset_path)
            img_env = lmdb.open(
                image_path, readonly=True, create=False, readahead=not False
            )
            self.img_txn = img_env.begin(buffers=True)
    
    def read_vsr_dataset(self,dataset_name, dataset_path = '../visual-spatial-reasoning/',splits_path='splits/', 
                         image_path = 'images/',sort = False, encode_labels = False):
        dataset = pd.read_json(dataset_path+splits_path+dataset_name, lines =True)
        dataset = dataset[['caption','image','label']]
        dataset.rename(columns = {'caption':'hypothesis', 'image':'Flickr30kID', 'label' : 'gold_label'}, inplace = True)
        if encode_labels:
            labels_encoding = {0:0,1:2}#leave the label 0 the same and convert 1 to 2 to mean entailment
            dataset['gold_label']=dataset['gold_label'].apply(lambda label: labels_encoding[label])
        if(dataset_name=='train.json'):
            dataset.drop(labels=[1786,3569,4553,4912], axis=0, inplace = True)
        elif(dataset_name=='test.json'):
            dataset.drop(labels=[135,614,1071,1621,1850], axis=0, inplace = True)
        elif(dataset_name=='dev.json'):
            dataset.drop(labels=[807], axis=0, inplace = True)
        dataset.reset_index(drop=True, inplace=True)
        if sort:
            dataset.sort_values(by="hypothesis", key=lambda x: x.str.len(), inplace = True)
        return dataset
    
    def read_dataset(self, url,sort = False):
        dataset = pd.read_csv(url)
        labels_encoding = {'contradiction':0,'neutral': 1,
                           'entailment':2}
        dataset = dataset[['hypothesis','Flickr30kID','gold_label']]
        dataset['gold_label']=dataset['gold_label'].apply(lambda label: labels_encoding[label])
        if sort:
            dataset.sort_values(by="hypothesis", key=lambda x: x.str.len(), inplace = True)
        return dataset
    
    def get_text_features(self,text): 
        #preprocess text
        inputs = self.lxmert_tokenizer(
            text,
            truncation=True,
            padding = True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        return inputs
    
    def __getitem__(self, idx):
        sample = self.dataset.loc[idx]
        img_name = sample['Flickr30kID']
        text = sample['hypothesis']
        label = sample['gold_label']
        item_img = pickle.loads(self.img_txn.get(img_name.encode()))
        
        item = {'text': text,
                'normalized_boxes': torch.tensor(item_img['normalized_boxes'][0], dtype = torch.float32),
                'features': torch.tensor(item_img['features'][0], dtype = torch.float32),
                'label': torch.tensor(label,dtype = torch.long)}
        return item
    
    def collate_fn(self,batch):
        #print(batch)
        text = [item['text'] for item in batch]
        inputs = self.get_text_features(text)
        item = {'input_ids': inputs['input_ids'].to(torch.int32),
                'attention_mask': inputs['attention_mask'].to(torch.int32),
                'token_type_ids': inputs['token_type_ids'].to(torch.int32),
                'label':torch.tensor([item['label'] for item in batch],dtype = torch.long),
                 'normalized_boxes': torch.stack(([item['normalized_boxes'] for item in batch])),
                'features': torch.stack(([item['features'] for item in batch]))
               }
        return item
        
        

    def __len__(self):
        return len(self.dataset.index)
    
    def __exit__(self):
        self.img_env.close()
        self.env.close()
        
max_length = 32
dataset = 'vsr'
lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

train = MyDataset("../e-ViL/data/esnlive_train.csv",
                      "./data/my_image_db", lxmert_tokenizer, 
                      max_length = max_length, vsr= 'train.json' if dataset=='vsr' else None)
test = MyDataset("../e-ViL/data/esnlive_test.csv",
                      "./data/my_image_db", lxmert_tokenizer, 
                      max_length = max_length,
                         vsr= 'test.json' if dataset=='vsr' else None)
dev = MyDataset("../e-ViL/data/esnlive_dev.csv",
                      "./data/my_image_db", lxmert_tokenizer, 
                      max_length = max_length, vsr= 'dev.json' if dataset=='vsr' else None)


def read_vsr_dataset(dataset_name, dataset_path = '../visual-spatial-reasoning/',splits_path='splits/', image_path = 'images/'):
    dataset = pd.read_json(dataset_path+splits_path+dataset_name, lines =True)
    dataset.rename(columns = {'caption':'hypothesis', 'image':'Flickr30kID', 'label' : 'gold_label'}, inplace = True)
    #dataset['Flickr30kID']=dataset['Flickr30kID'].apply(lambda img_name: dataset_path + image_path + img_name )
    if(dataset_name=='train.json'):
        dataset.drop(labels=[1786,3569,4553,4912], axis=0, inplace = True)
    elif(dataset_name=='test.json'):
        dataset.drop(labels=[135,614,1071,1621,1850], axis=0, inplace = True)
    elif(dataset_name=='dev.json'):
        dataset.drop(labels=[807], axis=0, inplace = True)
    dataset.reset_index(drop=True, inplace=True)
    return dataset

class MyDatasetContrastive(torch.utils.data.Dataset):
    def __init__(self,pandas_dataframe, lxmert_tokenizer, img_txn):
        self.lxmert_tokenizer = lxmert_tokenizer
        self.img_txn = img_txn
        self.dataset = pandas_dataframe

    def get_text_features(self,text): 
        inputs = self.lxmert_tokenizer(
            text,
            truncation=True,
            padding = True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        return inputs

    def __getitem__(self, idx):
        sample = self.dataset.loc[idx]
        #print(sample)
        img_name = sample['Flickr30kID']
        text = sample['hypothesis']
        label = sample['gold_label']
        contrast_label = sample['contrast_label']
        item_img = pickle.loads(self.img_txn.get(img_name.encode()))

        item = {'text': text,
                'normalized_boxes': torch.tensor(item_img['normalized_boxes'][0], dtype = torch.float32),
                'features': torch.tensor(item_img['features'][0], dtype = torch.float32),
                'label': torch.tensor(label,dtype = torch.long),
                'contrast_label': torch.tensor(contrast_label,dtype = torch.long)}
        return item

    def collate_fn(self,batch):
        text = [item['text'] for item in batch]
        inputs = self.get_text_features(text)
        item = {'input_ids': inputs['input_ids'].to(torch.int32),
                'attention_mask': inputs['attention_mask'].to(torch.int32),
                'token_type_ids': inputs['token_type_ids'].to(torch.int32),
                'label':torch.tensor([item['label'] for item in batch],dtype = torch.long),
                'contrast_label':torch.tensor([item['contrast_label'] for item in batch],dtype = torch.long),
                 'normalized_boxes': torch.stack(([item['normalized_boxes'] for item in batch])),
                'features': torch.stack(([item['features'] for item in batch]))
               }
        return item

    def __len__(self):
        return len(self.dataset.index)
    
train_dataset = read_vsr_dataset('train.json')
test_dataset = read_vsr_dataset('test.json')
dev_dataset = read_vsr_dataset('dev.json')
print(len(train_dataset))
print(len(test_dataset))
print(len(dev_dataset))

def get_nouns(phrase):
    is_noun = lambda pos: pos[:2] == 'NN'
    mwe = nltk.tokenize.MWETokenizer([('dining', 'table'), ('cell', 'phone'),('wine','glass'),('parking','meter'),
                                      ('hair','drier'),('fire','hydrant'),('traffic','light'),
                                     ('baseball','glove'),('sports','ball'),('stop','sign')], separator=' ')
    exceptions =['front','side','middle','top','part','edge']
    include = ['oven']
    tokens = nltk.word_tokenize(phrase)
    nouns = [word for (word, pos) in nltk.pos_tag(tokens) if is_noun(pos)]
    aggregated = mwe.tokenize(nouns)#get the compound names
    filtered_nouns = []
    for noun in aggregated:
        tag = nltk.pos_tag([noun])
        if (tag[0][1]=='NN' or tag[0][1]=='NNS' or tag[0][1]=='JJ') and noun not in exceptions or noun in include:
            filtered_nouns.append(noun)
    return filtered_nouns

def apply_entities(df):
    df['entities'] = df['hypothesis'].apply(lambda x: tuple(sorted(get_nouns(x))))

apply_entities(train_dataset)

group1 = {1:['on','on top of','at','above','over'], 0:['under','beneath','below','down from']}

group2 = {1: ['in front of','ahead of','past'], 0:['behind','at the back of','beyond']}

group3 = {1:['at the right side of','right of'],0: ['at the left side of','left of']}

group4 = {1:['in','inside','within','into','among','enclosed by','in the middle of','between'] ,
          0:['surrounding','around','contains','off','outside','out of']}

group5 = {1:['at the edge of','beside','near','close to','adjacent to',
              'attached to','by','next to','at the side of','congruent','connected to','touching','with'],
         0: ['far away from','far from','away from','detached from']}

group6 = {1:['facing away from','facing','toward','against'],0:['facing away from']}

group7 = {1:['parallel to','alongside','along'],
          0:['perpendicular to']}

group8 = {1:['across from','opposite to','across'],
          0:[]}

group9 = {1:['has a part','consists of'],
           0:['part of']}

groups = [group1,group2,group3,group4,group5,group6,group7,group8,group9]

#negatives_df['contrast_label'] = negatives_df['gold_label'].apply(lambda x : 1-x)#invert the label
def get_relation_datasets(df,groups):
    entities_set = set()
    for index, row in df.iterrows():
            entities = row['entities']
            entities_set.add(entities)#count the number of object pairs
    same_entities_same_label_diff_rel = {}
    for entities in list(entities_set):
        for group in groups:
            label = 1
            negatives = group[0]
            negatives_df = df[(df['relation'].isin(negatives)) & (df['gold_label'] == label) & (df['entities'] == entities)].reset_index(drop=True).copy()
            negatives_df['contrast_label'] = 0
            positives = group[1]
            positives_df = df[(df['relation'].isin(positives)) & (df['gold_label'] == label) & (df['entities'] == entities)].reset_index(drop=True).copy()
            positives_df['contrast_label'] = 1
            entities_group = tuple(sorted(positives + negatives))
            entities_dataset = pd.concat([positives_df,negatives_df]).reset_index()
            if(len(entities_dataset.index)>1):
                key1 = tuple([entities,entities_group,label])
                same_entities_same_label_diff_rel[key1]= entities_dataset
    return same_entities_same_label_diff_rel

relation_datasets = get_relation_datasets(train_dataset,groups)

relation_datasets_list = list(relation_datasets.values())
print(len(relation_datasets_list))

def get_caption_datasets(df):
    """
    Receives a Pandas dataframe and return a dictionary with the captions as keys and the
    datasets of that captions as values
    """
    #get a set with the captions
    captions_set = set()
    for index, row in df.iterrows():
            caption = row['hypothesis']
            captions_set.add(caption)
    #create a dataset for each caption in the set
    same_cap = {}
    for caption in list(captions_set):
        same_cap[caption]= df.loc[df.hypothesis.apply(lambda x: caption == x)].reset_index(drop=True)
        same_cap[caption]['contrast_label'] = same_cap[caption]['gold_label']
        if(len(same_cap[caption].index)<2):
            same_cap.pop(caption)
        
    return same_cap

caption_datasets = get_caption_datasets(train_dataset)

caption_datasets_list = list(caption_datasets.values())
print(len(caption_datasets_list))

from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.losses import GenericPairLoss
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

#https://github.com/KevinMusgrave/pytorch-metric-learning/issues/281
class SupConLoss(GenericPairLoss):
    def __init__(self, temperature, **kwargs):
        super().__init__(mat_based_loss=True, **kwargs)
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)

    def _compute_loss(self, mat, pos_mask, neg_mask):
        sim_mat = mat / self.temperature
        sim_mat_max, _ = sim_mat.max(dim=1, keepdim=True)
        sim_mat = sim_mat - sim_mat_max.detach()  # for numerical stability

        denominator = lmu.logsumexp(
            sim_mat, keep_mask=(pos_mask + neg_mask).bool(), add_one=False, dim=1
        )
        log_prob = sim_mat - denominator
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (
            pos_mask.sum(dim=1) + c_f.small_val(sim_mat.dtype)
        )
        losses = self.temperature * mean_log_prob_pos

        return {
            "loss": {
                "losses": -losses,
                "indices": c_f.torch_arange_from_size(sim_mat),
                "reduction_type": "element",
            }
        }

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def get_default_distance(self):
        return DotProductSimilarity()
    
from transformers import LxmertForPreTraining

class LxmertForBinaryClassification(torch.nn.Module):
    def __init__(self, lxmert):
        super(LxmertForBinaryClassification, self).__init__()
        self.lxmert = lxmert

    def forward(self, input_ids, attention_mask, features, normalized_boxes, token_type_ids, output_hidden_states = True,
                output_attentions= True, label=None, contrast_label = None):

        outputs = self.lxmert(
                input_ids = input_ids,
                attention_mask = attention_mask,
                visual_feats = features,
                visual_pos = normalized_boxes,
                token_type_ids = token_type_ids,
                output_hidden_states = output_hidden_states,
                output_attentions = output_attentions
                )
        outputs.logits = outputs["cross_relationship_score"]
        return outputs
    
    def predict(self,item):
      """
      item (n_examples x n_features)
      """
      scores = model(**item)  # (n_examples x n_classes)
      predicted_labels = scores.logits.argmax(dim=-1)  # (n_examples)
      return predicted_labels
    
    def save_model(self,path):
        torch.save(self.state_dict(), path)
        
    def load_model(self,path):
        self.load_state_dict(torch.load(path))
        
class MyEvaluator():
  def __init__(self,model,test, device = None):
    self.test_dataset = test
    self.model = model
    self.device = device
  
  def evaluate(self, batch_size = 8):
      self.model.eval()
      loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle = False, num_workers = 4,
                         collate_fn = self.test_dataset.collate_fn)
      n_correct = 0
      n_possible = 0
      for item in loader:
        item['input_ids']=item['input_ids'].to(self.device)
        item['attention_mask']= item['attention_mask'].to(self.device)
        item['token_type_ids']= item['token_type_ids'].to(self.device)
        item['normalized_boxes'] = item['normalized_boxes'].to(self.device)
        item['features']= item['features'].to(self.device)
        item['label'] = item['label'].to(self.device)
        y_hat = self.model.predict(item)
        y = item['label']
        n_correct += (y == y_hat).sum().item()
        n_possible += float(y.shape[0])
      self.model.train()
      return n_correct / n_possible
    
class MyContrastiveTrainer():
    def __init__(self,model,train_list ,eval_test, device = None, num_labels = 2, batch_size = 32):
        self.device = device
        self.model = model
        self.train_list = train_list
        self.eval_test = eval_test
        self.test_acc_list = []
        self.num_labels = num_labels
        self.loss_fct_contrastive = SupConLoss(0.07)
        self.loss_fct_cross_entropy = torch.nn.CrossEntropyLoss()
        self.output_loss = lambda features,contrast_labels, labels, output : self.loss_fct_contrastive(features, contrast_labels) + self.loss_fct_cross_entropy(output.logits, labels)
        self.total_steps = len(train_list)
        self.train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = 4,
                                     collate_fn = dataset.collate_fn) for dataset in train_list]
        
        
    def train_model(self,batch_size = None, lr= None, epochs=None):
        optimizer = AdamW(self.model.parameters(), lr=lr)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps= epochs * self.total_steps
        )
        for epoch in range(epochs):
            progress_bar = tqdm(range(math.ceil(self.total_steps)))
            train_losses = []
            for dataloader in self.train_loaders:
                for item in dataloader:
                    item['input_ids']=item['input_ids'].to(self.device)
                    item['attention_mask']= item['attention_mask'].to(self.device)
                    item['token_type_ids']= item['token_type_ids'].to(self.device)
                    item['normalized_boxes'] = item['normalized_boxes'].to(self.device)
                    item['features']= item['features'].to(self.device)
                    item['label'] = item['label'].to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model.forward(**item, output_hidden_states = True, output_attentions= True)
                    label = item['label']
                    contrast_label = item['contrast_label']
                    visual_output = outputs['vision_hidden_states'][-1]
                    lang_output = outputs['language_hidden_states'][-1]
                    pooled_output = self.model.lxmert.lxmert.pooler(lang_output)
                    features = torch.cat((flatten(visual_output),pooled_output),dim=1)
                    loss = self.output_loss(features, contrast_label, label, outputs)
                    train_losses.append(loss)
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    progress_bar.update(1)
            test_acc = self.eval_test.evaluate(batch_size = batch_size)
            self.test_acc_list.append(test_acc)
            print('--- Epoch ',epoch,' Acc: ',test_acc)
            mean_loss = torch.tensor(train_losses).mean().item()
            print('Training loss: %.4f' % (mean_loss))
        return
    
vsr_image_path = '../lxmert/data/vsr-images'
img_env = lmdb.open(vsr_image_path, readonly=True, create=False, readahead=not False)
img_txn = img_env.begin(buffers=True)

all_datasets = relation_datasets_list #caption_datasets_list
contrastive_datasets = []

for pandas_dataframe in all_datasets:
    contrastive_datasets.append(MyDatasetContrastive(pandas_dataframe, lxmert_tokenizer, img_txn))

print(len(contrastive_datasets))

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = LxmertForBinaryClassification(LxmertForPreTraining.from_pretrained('unc-nlp/lxmert-base-uncased'))
model = model.to(device)

model.load_model('../lxmert/vsr_len32_batch32_lr1e-05')

batch_size = 32
dev_evaluator = MyEvaluator(model, dev, device = device)
dev_acc = dev_evaluator.evaluate(batch_size = batch_size)
print(dev_acc)

epochs = 5
batch_size = 32
lr = 1e-5
test_evaluator = MyEvaluator(model, test, device = device)
dev_evaluator = MyEvaluator(model, dev, device = device)
#trainer = MyTrainer(model,train,test_evaluator, device = device, num_labels = num_labels)
trainer = MyContrastiveTrainer(model, contrastive_datasets, test_evaluator, device = device, num_labels = 2)
print("-----Training Model-----")
trainer.train_model(epochs=epochs ,batch_size = batch_size, lr = lr)
print('----Training finished-----')
dev_acc = dev_evaluator.evaluate(batch_size = batch_size)
print("---- Dev Acc: ",dev_acc)
train_acc = MyEvaluator(model,train,device=device).evaluate(batch_size = batch_size)
print("--- Train Acc: ", train_acc)
model.save_model('contrastive_model')