#!/usr/bin/env python
import numpy as np
from ctypes import c_void_p
import mlsl
import theano
import theano.tensor as T
from theano.contrib.mlsl.mlsl_api import AllReduce, addr
'''
def addr(x):
    xaddr, offset = x.ctypes.data_as(c_void_p), 0
    for i in range(len(x.shape)):
        if x.strides[i] < 0: offset += (x.shape[i]-1)*x.strides[i]
    xaddr.value += offset
    return xaddr

# For data parallelism
class AllReduce(theano.Op):
    def __init__(self,dist, count):
        self.dist = dist
        self.count = count

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        outputs = [x.type(),]
        return theano.Apply(self,[x],outputs)

    def perform(self,node,inputs,outputs):
        # In place
        dist = self.dist
        x, = inputs
        y, = outputs
        #print 'in perform ',x.shape
        y[0]=x
        req=dist.all_reduce(send_buf=addr(x), recv_buf=addr(x), count=self.count, data_type=0, red_type=0, group_type=2)
        mlsl_obj.wait(req)

    def grad(self,inputs,grads):
        return [theano.gradient.DisconnectedType()(),]
'''
if __name__ == '__main__':
	mlsl_obj = mlsl.MLSL()
	mlsl_obj.init()
	node_idx = mlsl_obj.get_process_idx()
	node_num = mlsl_obj.get_process_count()
	print 'rank ', node_idx
	print 'nodes ', node_num
	dist=mlsl_obj.create_distribution(node_num,1)

	# This is theano style mlsl test 
	from theano import shared
	state = shared(0)
	inc = T.iscalar('inc')
	reduce_op=AllReduce(mlsl_obj,dist,1)
	accumulator = theano.function([inc], state, updates=[(state, state+reduce_op(inc))])
	if node_idx == 0:
		tmp = 100
	else:
		tmp = 200
	accumulator(tmp/node_num)
	# The output should be 150
	print state.get_value()
	
	# This is raw mlsl test
	a=np.empty((2),dtype=np.float32)
	b=np.empty((2),dtype=np.float32)
	if mlsl_obj.get_process_idx() == 0:
		a[0]=1.
		a[1]=3.
	else:
		a[0]=3.
		a[1]=4.	
	print 'a ', a
	print 'a address ',addr(a)
	req=dist.all_reduce(send_buf=addr(a), recv_buf=addr(b), count=2, data_type=0, red_type=0, group_type=2)
	mlsl_obj.wait(req)
	print 'b ',b/mlsl_obj.get_process_count()

	if mlsl_obj.get_process_idx() == 0:
		b[0]=19.
		b[1]=32.
	else:
		b[0]=13.
		b[1]=42.	
	req=dist.all_reduce(send_buf=addr(b), recv_buf=addr(b), count=2, data_type=0, red_type=0, group_type=2)
	mlsl_obj.wait(req)
	print 'b ',b/mlsl_obj.get_process_count()
	
	mlsl_obj.delete_distribution(dist)
	mlsl_obj.finalize()
