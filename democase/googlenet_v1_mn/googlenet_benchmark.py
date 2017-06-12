import theano
import theano.tensor as T
import numpy as np
from googlenet_theano import googlenet, compile_train_model, compile_val_model, set_learning_rate
import time
from datetime import datetime
import sys
import traceback
import theano.sandbox.mlsl as mlsl
#sys.path.append('/home/zhaopeng/smallcase/theano/sandbox/mlsl')
#sys.path.append('/data/lfs/user/patric/huawei/tools/theano/theano/sandbox/mlsl/')

try:
    #import distributed as distributed
    import multinode as distributed
    print('mlsl is imported')
except ImportError as e:
    print ('Failed to import distributed module, please double check')
    print(traceback.format_exc())
#    sys.exit(0)

def time_theano_run(func, info_string):
    num_batches = 50
    num_steps_burn_in = 10
    durations = []
    for i in xrange(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = func()
        duration = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                print('%s: Iteration %d, %s, time: %.2f ms' %
                      (datetime.now(), i - num_steps_burn_in, info_string, duration*1000))
            durations.append(duration)
    durations = np.array(durations)
    print('%s: Average %s pass: %.2f ms ' %
          (datetime.now(), info_string, durations.mean()*1000))

def googlenet_train(batch_size=256, image_size=(3, 224, 224)):

    input_shape = (batch_size,) + image_size
    model = googlenet(input_shape)

    dist = distributed.Distribution()
    print ('dist.rank: ', dist.rank, 'dist.size: ', dist.size)

    #rank==1: single mode; else multinode
    if dist.size <= 1:
    	  print("Run single mode")
    	  multinode_mode = False
    else:
        print("Run multinode mode")
        multinode_mode = True	
        
    if multinode_mode == True:      
        distributed.set_global_batch_size(batch_size * dist.size)
        distributed.set_param_count(len(model.params))

    (train_model, shared_x, shared_y, shared_lr) = compile_train_model(model, batch_size = batch_size,multinode_mode = multinode_mode)
    (validate_model, shared_x, shared_y) = compile_val_model(model, batch_size = batch_size)

    images = np.random.random_integers(0, 255, input_shape).astype('float32')
    labels = np.random.random_integers(0, 999, batch_size).astype('int32')
    shared_x.set_value(images)
    shared_y.set_value(labels)
    #set_learning_rate(shared_lr, iter)

#    print("Run model validation: dropout off")
    model.set_dropout_off()
    time_theano_run(validate_model, 'Forward')

#    print("Run model training dropout on")
    model.set_dropout_on()
    time_theano_run(train_model, 'Forward-Backward')

    dist.destroy()

if __name__ =='__main__':
    googlenet_train(batch_size=32)
    #googlenet_train(batch_size=16)
