import os
import argparse
import logging
import sys
import tensorflow as tf
import numpy as np

sys.path.append('../../vendor/Joint-NLU/src')
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from utils import createVocabulary
from utils import loadVocabulary
from utils import computeF1Score
from utils import DataProcessor

DATASETS_ROOT = '../../dataset'

parser = argparse.ArgumentParser(allow_abbrev=False)

#Network
parser.add_argument("--num_units", type=int, default=128, help="Network size.", dest='layer_size')
parser.add_argument("--model_type", type=str, default='full', help="""full(default) | intent_only
                                                                    full: full attention model
                                                                    intent_only: intent attention model""")

#Training Environment
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--max_epochs", type=int, default=20, help="Max epochs to train.")
parser.add_argument("--no_early_stop", action='store_false',dest='early_stop', help="Disable early stop, which is based on sentence level accuracy.")
parser.add_argument("--patience", type=int, default=5, help="Patience to wait before stop.")

#Model and Vocab
parser.add_argument("--dataset", type=str, default=None, help="""Type 'atis' or 'snips' to use dataset provided by us or enter what ever you named your own dataset.
                Note, if you don't want to use this part, enter --dataset=''. It can not be None""")
parser.add_argument("--model_path", type=str, default='./src/model', help="Path to save model.")
parser.add_argument("--vocab_path", type=str, default='./src/vocab', help="Path to vocabulary files.")

#Data
parser.add_argument("--train_data_path", type=str, default='train', help="Path to training data files.")
parser.add_argument("--test_data_path", type=str, default='test', help="Path to testing data files.")
parser.add_argument("--valid_data_path", type=str, default='valid', help="Path to validation data files.")
parser.add_argument("--input_file", type=str, default='seq.in', help="Input file name.")
parser.add_argument("--slot_file", type=str, default='seq.out', help="Slot file name.")
parser.add_argument("--intent_file", type=str, default='label', help="Intent file name.")

arg=parser.parse_args(['--dataset', 'joint-nlu'])


#Print arguments
for k,v in sorted(vars(arg).items()):
    print(k,'=',v)
print()

if arg.model_type == 'full':
    add_final_state_to_intent = True
    remove_slot_attn = False
elif arg.model_type == 'intent_only':
    add_final_state_to_intent = True
    remove_slot_attn = True
else:
    print('unknown model type!')
    exit(1)

#full path to data will be: ./data + dataset + train/test/valid
if arg.dataset == None:
    print('name of dataset can not be None')
    exit(1)
elif arg.dataset == 'snips':
    print('use snips dataset')
elif arg.dataset == 'atis':
    print('use atis dataset')
else:
    print('use own dataset: ',arg.dataset)

full_train_path = os.path.join(DATASETS_ROOT,arg.dataset,arg.train_data_path)
full_test_path = os.path.join(DATASETS_ROOT,arg.dataset,arg.test_data_path)
full_valid_path = os.path.join(DATASETS_ROOT,arg.dataset,arg.valid_data_path)

createVocabulary(os.path.join(full_train_path, arg.input_file), os.path.join(DATASETS_ROOT, arg.dataset, 'in_vocab'))
createVocabulary(os.path.join(full_train_path, arg.slot_file), os.path.join(DATASETS_ROOT, arg.dataset, 'slot_vocab'))
createVocabulary(os.path.join(full_train_path, arg.intent_file), os.path.join(DATASETS_ROOT, arg.dataset, 'intent_vocab'))

in_vocab = loadVocabulary(os.path.join(DATASETS_ROOT, arg.dataset, 'in_vocab'))
slot_vocab = loadVocabulary(os.path.join(DATASETS_ROOT, arg.dataset, 'slot_vocab'))
intent_vocab = loadVocabulary(os.path.join(DATASETS_ROOT, arg.dataset, 'intent_vocab'))

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units, kernel_regularizer = tf.keras.regularizers.l2(0.02))
        self.W2 = tf.keras.layers.Dense(units,kernel_regularizer = tf.keras.regularizers.l2(0.02))
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)


        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# In[ ]:


#both query and values are time distributed
class CustomBahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(CustomBahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units,kernel_regularizer = tf.keras.regularizers.l2(0.02))
        self.W2 = tf.keras.layers.Conv1D(units,5,1,'same')
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        #hidden_with_time_axis = tf.expand_dims(query, 1)
        hidden_with_time_axis = query

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        #context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# In[ ]:


class SlotGate(tf.keras.Model):
    def __init__(self, units):
        super(SlotGate, self).__init__()
        self.W1 = tf.keras.layers.Dense(units, kernel_regularizer = tf.keras.regularizers.l2(0.02))
        self.W2 = tf.keras.layers.Dense(units, kernel_regularizer = tf.keras.regularizers.l2(0.02))
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)
        
        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        #context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# In[ ]:


from tensorflow.keras import layers
class CustomDropout(layers.Layer):
  def __init__(self, rate, **kwargs):
    super(CustomDropout, self).__init__(**kwargs)
    self.rate = rate

  def call(self, inputs, training=None):
    if training:
        return tf.nn.dropout(inputs, rate=self.rate)
    return inputs
  
class BaselineModel(tf.keras.Model):
  def __init__(self, input_size, slot_size, intent_size, layer_size = 128):
    super(BaselineModel, self).__init__()
    self.embedding = tf.keras.layers.Embedding(input_size, layer_size)
    self.bilstm = tf.keras.layers.Bidirectional(CuDNNLSTM(layer_size, return_sequences=True,return_state=True))
    self.dropout = CustomDropout(0.5)
    self.intent_out = tf.keras.layers.Dense(intent_size, activation=None)
    self.slot_out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(slot_size, activation=None))
  
  @tf.function  
  def call(self, inputs, sequence_length, isTraining=True):
    x = self.embedding(inputs)
    state_outputs, forward_h, forward_c, backward_h, backward_c = self.bilstm(x)
    
    state_outputs = self.dropout(state_outputs, isTraining)
    forward_h = self.dropout(forward_h, isTraining)
    backward_h = self.dropout(backward_h, isTraining)
   
    final_state = tf.keras.layers.concatenate([forward_h,backward_h])
    intent = self.intent_out(final_state)
    slots = self.slot_out(state_outputs)
    outputs = [slots, intent]
    return outputs
    


# In[ ]:


class SlotGatedModel(tf.keras.Model):
  def __init__(self, input_size, slot_size, intent_size, layer_size = 128):
    super(SlotGatedModel, self).__init__()
    self.embedding = tf.keras.layers.Embedding(input_size, layer_size)
    self.bilstm = tf.keras.layers.Bidirectional(CuDNNLSTM(layer_size, return_sequences=True,return_state=True))
    self.dropout = CustomDropout(0.5)
    
    self.attn_size = 2*layer_size
    self.slot_att = CustomBahdanauAttention(self.attn_size)
    self.intent_att = BahdanauAttention(self.attn_size)
    self.slot_gate = SlotGate(self.attn_size)
    
    self.intent_out = tf.keras.layers.Dense(intent_size, activation=None)
    self.slot_out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(slot_size, activation=None))
  
  @tf.function  
  def call(self, inputs, sequence_length, isTraining=True):
    x = self.embedding(inputs)
    state_outputs, forward_h, forward_c, backward_h, backward_c = self.bilstm(x)
    
    state_outputs = self.dropout(state_outputs, isTraining)
    forward_h = self.dropout(forward_h, isTraining)
    backward_h = self.dropout(backward_h, isTraining)
    final_state = tf.keras.layers.concatenate([forward_h,backward_h])
    
    slot_d, _ = self.slot_att(state_outputs, state_outputs)
    intent_d, _ = self.intent_att(final_state, state_outputs)
    
    intent_fd = tf.keras.layers.concatenate([intent_d,final_state], -1)
    slot_gated,_ = self.slot_gate(intent_fd, slot_d)
    slot_fd = tf.keras.layers.concatenate([slot_gated,state_outputs], -1)
    
    
    intent = self.intent_out(intent_fd)
    slots = self.slot_out(slot_fd)
    outputs = [slots, intent]
    return outputs


# In[ ]:


tf.keras.backend.clear_session()
model = SlotGatedModel(len(in_vocab['vocab']), len(slot_vocab['vocab']), len(intent_vocab['vocab']), layer_size=arg.layer_size)


# In[ ]:


def valid(in_path, slot_path, intent_path):
    data_processor_valid = DataProcessor(in_path, slot_path, intent_path, in_vocab, slot_vocab, intent_vocab)
    pred_intents = []
    correct_intents = []
    slot_outputs = []
    correct_slots = []
    input_words = []

    #used to gate
    #gate_seq = []
    while True:
        in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq = data_processor_valid.get_batch(arg.batch_size)
        #feed_dict = {input_data.name: in_data, sequence_length.name: length}
        #ret = sess.run(inference_outputs, feed_dict)
        slots, intent = model(in_data, length, isTraining = False)
        for i in np.array(intent):
            pred_intents.append(np.argmax(i))
        for i in intents:
            correct_intents.append(i)

        pred_slots = slots
        for p, t, i, l in zip(pred_slots, slot_data, in_data, length):
            p = np.argmax(p, 1)
            tmp_pred = []
            tmp_correct = []
            tmp_input = []
            for j in range(l):
                tmp_pred.append(slot_vocab['rev'][p[j]])
                tmp_correct.append(slot_vocab['rev'][t[j]])
                tmp_input.append(in_vocab['rev'][i[j]])

            slot_outputs.append(tmp_pred)
            correct_slots.append(tmp_correct)
            input_words.append(tmp_input)

        if data_processor_valid.end == 1:
            break

    pred_intents = np.array(pred_intents)
    correct_intents = np.array(correct_intents)
    accuracy = (pred_intents==correct_intents)
    semantic_error = accuracy
    accuracy = accuracy.astype(float)
    accuracy = np.mean(accuracy)*100.0

    index = 0
    for t, p in zip(correct_slots, slot_outputs):
        # Process Semantic Error
        if len(t) != len(p):
            raise ValueError('Error!!')

        for j in range(len(t)):
            if p[j] != t[j]:
                semantic_error[index] = False
                break
        index += 1
    semantic_error = semantic_error.astype(float)
    semantic_error = np.mean(semantic_error)*100.0

    f1, precision, recall = computeF1Score(correct_slots, slot_outputs)
    print('slot f1: ' + str(f1) + '\tintent accuracy: ' + str(accuracy) + '\tsemantic_error: ' + str(semantic_error))
    #print('intent accuracy: ' + str(accuracy))
    #print('semantic error(intent, slots are all correct): ' + str(semantic_error))

    data_processor_valid.close()
    return f1,accuracy,semantic_error,pred_intents,correct_intents,slot_outputs,correct_slots,input_words#,gate_seq

  


# In[ ]:


#training
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
ckpt = tf.train.Checkpoint(step=tf.Variable(-1), optimizer=opt, net=model)
manager = tf.train.CheckpointManager(ckpt, arg.model_path + "/" + arg.dataset + '/tf_ckpts', max_to_keep=3)
data_processor = None
valid_err = 0
no_improve= 0
save_path = os.path.join(arg.model_path , str(arg.dataset) + "/")
for epoch in range(50):
    while True:
        if data_processor == None:
            i_loss = 0
            s_loss = 0
            batches = 0
            data_processor = DataProcessor(os.path.join(full_train_path, arg.input_file), os.path.join(full_train_path, arg.slot_file), os.path.join(full_train_path, arg.intent_file), in_vocab, slot_vocab, intent_vocab)
        in_data, slot_labels, slot_weights, length, intent_labels,in_seq,_,_ = data_processor.get_batch(arg.batch_size)
        
        with tf.GradientTape() as tape:
          slots, intent = model(in_data, length, isTraining = True)
          intent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=intent_labels, logits=intent)  
          #slot_loss
          slots_out = tf.reshape(slots, [-1,len(slot_vocab['vocab'])])
          slots_shape = tf.shape(slot_labels)
          slot_reshape = tf.reshape(slot_labels, [-1])
          crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=slot_reshape, logits=slots_out)
          crossent = tf.reshape(crossent, slots_shape)
          slot_loss = tf.reduce_sum(crossent*slot_weights, 1)
          total_size = tf.reduce_sum(slot_weights, 1)
          total_size += 1e-12
          slot_loss = slot_loss / total_size
          
          total_loss = intent_loss + slot_loss + tf.reduce_sum(model.losses)
        
        grads = tape.gradient(total_loss, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))
        s_loss = s_loss + tf.reduce_sum(slot_loss)/tf.cast(arg.batch_size, tf.float32)
        i_loss = i_loss +  tf.reduce_sum(intent_loss)/tf.cast(arg.batch_size, tf.float32)
        batches = batches + 1
        if data_processor.end == 1:
            data_processor.close()
            data_processor = None
            break
            
    #print("Training Epoch: " ,epoch," Slot Loss: ",s_loss/batches, " Intent_Loss: ", i_loss/batches)
    print("EPOCH: ", epoch, " *******************************************************************")
    print('Train:', end="\t")
    _ = valid(os.path.join(full_train_path, arg.input_file), os.path.join(full_train_path, arg.slot_file), os.path.join(full_train_path, arg.intent_file))
    
    print('Valid:', end="\t")
    epoch_valid_slot, epoch_valid_intent, epoch_valid_err,valid_pred_intent,valid_correct_intent,valid_pred_slot,valid_correct_slot,valid_words = valid(os.path.join(full_valid_path, arg.input_file), os.path.join(full_valid_path, arg.slot_file), os.path.join(full_valid_path, arg.intent_file))

    print('Test:', end="\t")
    epoch_test_slot, epoch_test_intent, epoch_test_err,test_pred_intent,test_correct_intent,test_pred_slot,test_correct_slot,test_words = valid(os.path.join(full_test_path, arg.input_file), os.path.join(full_test_path, arg.slot_file), os.path.join(full_test_path, arg.intent_file))
    
    ckpt.step.assign_add(1)
    if epoch_valid_err <= valid_err:
        no_improve += 1
    else:
        valid_err = epoch_valid_err
        no_improve = 0
        print("Saving", str(ckpt.step), "with valid accuracy:", valid_err   )
        save_path = manager.save()

    if arg.early_stop == True:
        if no_improve > arg.patience:
            print("EARLY BREAK")
            break


# In[ ]:


model.summary()


# In[ ]:


#Let's try to clear slate and reload maodel  .....
import time
tf.keras.backend.clear_session()
if arg.dataset == 'atis':
    test_batch = 893
elif arg.dataset == 'snips':
    test_batch = 700

model = MyModel(len(in_vocab['vocab']), len(slot_vocab['vocab']), len(intent_vocab['vocab']), layer_size=arg.layer_size)
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

ckpt = tf.train.Checkpoint(step=tf.Variable(-1), optimizer=opt, net=model)
manager = tf.train.CheckpointManager(ckpt, arg.model_path + "/" + arg.dataset + '/tf_ckpts', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

data_processor_test = DataProcessor(os.path.join(full_test_path, arg.input_file), os.path.join(full_test_path, arg.slot_file), os.path.join(full_test_path, arg.intent_file), in_vocab, slot_vocab, intent_vocab)
in_data, slot_labels, slot_weights, length, intent_labels,in_seq,_,_ = data_processor_test.get_batch(test_batch)
data_processor_test.close()



t = time.perf_counter()
slots, intent = model(in_data, length, isTraining = False)
elapsed = time.perf_counter() - t
print("Milli seconds per query:", (elapsed*1000)/float(100.0))

pred_intents = []
correct_intents = []
slot_outputs = []
correct_slots = []
input_words = []

for i in np.array(intent):
    pred_intents.append(np.argmax(i))
for i in intent_labels:
    correct_intents.append(i)

pred_slots = slots
for p, t, i, l in zip(pred_slots, slot_labels, in_data, length):
    p = np.argmax(p, 1)
    tmp_pred = []
    tmp_correct = []
    tmp_input = []
    for j in range(l):
        tmp_pred.append(slot_vocab['rev'][p[j]])
        tmp_correct.append(slot_vocab['rev'][t[j]])
        tmp_input.append(in_vocab['rev'][i[j]])

    slot_outputs.append(tmp_pred)
    correct_slots.append(tmp_correct)
    input_words.append(tmp_input)

pred_intents = np.array(pred_intents)
correct_intents = np.array(correct_intents)
accuracy = (pred_intents==correct_intents)
semantic_error = accuracy
accuracy = accuracy.astype(float)
accuracy = np.mean(accuracy)*100.0

index = 0
for t, p in zip(correct_slots, slot_outputs):
    # Process Semantic Error
    if len(t) != len(p):
        raise ValueError('Error!!')

    for j in range(len(t)):
        if p[j] != t[j]:
            semantic_error[index] = False
            break
    index += 1
semantic_error = semantic_error.astype(float)
semantic_error = np.mean(semantic_error)*100.0

f1, precision, recall = computeF1Score(correct_slots, slot_outputs)
print('slot f1: ' + str(f1) + '\tintent accuracy: ' + str(accuracy) + '\tsemantic_accuracy: ' + str(semantic_error))


# In[ ]:





# In[ ]:




